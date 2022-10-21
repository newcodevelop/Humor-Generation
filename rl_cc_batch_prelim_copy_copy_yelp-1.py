from email.mime import base
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torchtext
from torchtext.legacy.data import Field, BucketIterator
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.distributions import Categorical
import spacy
import numpy as np
import torch.nn.functional as F
import random
import math
import time
import resource
from sacrebleu import sentence_bleu
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import tensorflow as tf
# https://ai.stackexchange.com/questions/2405/how-do-i-handle-negative-rewards-in-policy-gradients-with-the-cross-entropy-loss

import tensorflow_probability as tfp_
tfd = tfp_.distributions
tfkl = tf.keras.layers
tfp = tfp_.layers




def reduce_logmeanexp_nodiag(x, axis=None):
  batch_size = x.shape[0]
  logsumexp = tf.reduce_logsumexp(x - tf.linalg.tensor_diag(np.inf * tf.ones(batch_size)), axis=axis)
  if axis:
    num_elem = batch_size - 1.
  else:
    num_elem  = batch_size * (batch_size - 1.)
  return logsumexp - tf.math.log(num_elem)

def tuba_lower_bound(scores, log_baseline=None):
  if log_baseline is not None:
    scores -= log_baseline[:, None]
  batch_size = tf.cast(scores.shape[0], tf.float32)
  # First term is an expectation over samples from the joint,
  # which are the diagonal elmements of the scores matrix.
  joint_term = tf.reduce_mean(tf.linalg.diag_part(scores))
  # Second term is an expectation over samples from the marginal,
  # which are the off-diagonal elements of the scores matrix.
  marg_term = tf.exp(reduce_logmeanexp_nodiag(scores))
  return 1. + joint_term -  marg_term

def nwj_lower_bound(scores):
  # equivalent to: tuba_lower_bound(scores, log_baseline=1.)
  return tuba_lower_bound(scores - 1.) 



def infonce_lower_bound(scores):
  """InfoNCE lower bound from van den Oord et al. (2018)."""
  nll = tf.reduce_mean(tf.linalg.diag_part(scores) - tf.reduce_logsumexp(scores, axis=1))
  # Alternative implementation:
  # nll = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=tf.range(batch_size))
  mi = tf.math.log(tf.cast(scores.shape[0], tf.float32)) + nll
  return mi




def log_interpolate(log_a, log_b, alpha_logit):
  """Numerically stable implementation of log(alpha * a + (1-alpha) * b)."""
  log_alpha = -tf.nn.softplus(-alpha_logit)
  log_1_minus_alpha = -tf.nn.softplus(alpha_logit)
  y = tf.reduce_logsumexp(tf.stack((log_alpha + log_a, log_1_minus_alpha + log_b)), axis=0)
  return y

def compute_log_loomean(scores):
  """Compute the log leave-one-out mean of the exponentiated scores.

  For each column j we compute the log-sum-exp over the row holding out column j.
  This is a numerically stable version of:
  log_loosum = scores + tfp.math.softplus_inverse(tf.reduce_logsumexp(scores, axis=1, keepdims=True) - scores) 
  Implementation based on tfp.vi.csiszar_divergence.csiszar_vimco_helper.
  """
  max_scores = tf.reduce_max(scores, axis=1, keepdims=True)
  lse_minus_max = tf.reduce_logsumexp(scores - max_scores, axis=1, keepdims=True)
  d = lse_minus_max + (max_scores - scores)
  d_ok = tf.not_equal(d, 0.)
  safe_d = tf.where(d_ok, d, tf.ones_like(d))
  loo_lse = scores + tfp_.math.softplus_inverse(safe_d)
  # Normalize to get the leave one out log mean exp
  loo_lme = loo_lse - tf.math.log(scores.shape[1] - 1.)
  return loo_lme

def interpolated_lower_bound(scores, baseline, alpha_logit):
  """Interpolated lower bound on mutual information.

  Interpolates between the InfoNCE baseline ( alpha_logit -> -infty),
  and the single-sample TUBA baseline (alpha_logit -> infty)

  Args:
    scores: [batch_size, batch_size] critic scores
    baseline: [batch_size] log baseline scores
    alpha_logit: logit for the mixture probability

  Returns:
    scalar, lower bound on MI
  """
  batch_size = scores.shape[0]
  # Compute InfoNCE baseline
  nce_baseline = compute_log_loomean(scores)
  # Inerpolated baseline interpolates the InfoNCE baseline with a learned baseline
  interpolated_baseline = log_interpolate(
      nce_baseline, tf.tile(baseline[:, None], (1, batch_size)), alpha_logit)
  # Marginal term.
  critic_marg = scores - tf.linalg.diag_part(interpolated_baseline)[:, None]
  marg_term = tf.exp(reduce_logmeanexp_nodiag(critic_marg))

  # Joint term.
  critic_joint = tf.linalg.diag_part(scores)[:, None] - interpolated_baseline
  joint_term = (tf.reduce_sum(critic_joint) -
                tf.reduce_sum(tf.linalg.diag_part(critic_joint))) / (batch_size * (batch_size - 1.))
  return 1 + joint_term  - marg_term



def js_fgan_lower_bound(f):
  """Lower bound on Jensen-Shannon divergence from Nowozin et al. (208)."""
  f_diag = tf.linalg.tensor_diag_part(f)
  first_term = tf.reduce_mean(-tf.nn.softplus(-f_diag))
  n = tf.cast(f.shape[0], tf.float32)
  second_term = (tf.reduce_sum(tf.nn.softplus(f)) - tf.reduce_sum(tf.nn.softplus(f_diag))) / (n * (n - 1.))
  return first_term - second_term

def js_lower_bound(f):
  """NWJ lower bound on MI using critic trained with Jensen-Shannon.

  The returned Tensor gives MI estimates when evaluated, but its gradients are
  the gradients of the lower bound of the Jensen-Shannon divergence."""
  js = js_fgan_lower_bound(f)
  mi = nwj_lower_bound(f)
  return js + tf.stop_gradient(mi - js)




def reduce_logmeanexp_nodiag_torch(x, axis=None):
  batch_size = x.size()[0]
  logsumexp = torch.logsumexp(x-torch.diag(np.inf*torch.ones(batch_size)).to('cuda'),dim=[0,1])
  
  if axis:
    num_elem = batch_size - 1.
  else:
    num_elem  = batch_size * (batch_size - 1.)
  return logsumexp - torch.log(torch.tensor(num_elem)).to('cuda')

def tuba_lower_bound_torch(scores, log_baseline=None):
  if log_baseline is not None:
    
    scores -= torch.tensor(log_baseline).unsqueeze(dim=-1).to('cuda')

  

  batch_size = scores.size()[0]
  # First term is an expectation over samples from the joint,
  # which are the diagonal elmements of the scores matrix.
  joint_term = torch.diagonal(scores).mean()
  # Second term is an expectation over samples from the marginal,
  # which are the off-diagonal elements of the scores matrix.
  marg_term = torch.exp(reduce_logmeanexp_nodiag_torch(scores))
  return 1. + joint_term -  marg_term

def nwj_lower_bound_torch(scores):
  return tuba_lower_bound_torch(scores, log_baseline=1.)
  # return tuba_lower_bound_torch(scores - 1.) 


# def estimate_mutual_information(estimator, x, y, critic_fn,
#                                 baseline_fn=None, alpha_logit=None):
#   """Estimate variational lower bounds on mutual information.

#   Args:
#     estimator: string specifying estimator, one of:
#       'nwj', 'infonce', 'tuba', 'js', 'interpolated'
#     x: [batch_size, dim_x] Tensor
#     y: [batch_size, dim_y] Tensor
#     critic_fn: callable that takes x and y as input and outputs critic scores
#       output shape is a [batch_size, batch_size] matrix
#     baseline_fn (optional): callable that takes y as input 
#       outputs a [batch_size]  or [batch_size, 1] vector
#     alpha_logit (optional): logit(alpha) for interpolated bound

#   Returns:
#     scalar estimate of mutual information
#   """
#   scores = critic_fn(x, y)
#   if baseline_fn is not None:
#     # Some baselines' output is (batch_size, 1) which we remove here.
#     log_baseline = tf.squeeze(baseline_fn(y))
#   if estimator == 'infonce':
#     mi = infonce_lower_bound(scores)
#   elif estimator == 'nwj':
#     mi = nwj_lower_bound(scores)
#   elif estimator == 'tuba':
#     mi = tuba_lower_bound(scores, log_baseline)
#   elif estimator == 'js':
#     mi = js_lower_bound(scores)
#   elif estimator == 'interpolated':
#     assert alpha_logit is not None, "Must specify alpha_logit for interpolated bound."
#     mi = interpolated_lower_bound(scores, log_baseline, alpha_logit)
#   return mi


def estimate_mutual_information_torch(estimator, x, y, critic_fn,
                                baseline_fn=None, alpha_logit=None):
  """Estimate variational lower bounds on mutual information.

  Args:
    estimator: string specifying estimator, one of:
      'nwj', 'infonce', 'tuba', 'js', 'interpolated'
    x: [batch_size, dim_x] Tensor
    y: [batch_size, dim_y] Tensor
    critic_fn: callable that takes x and y as input and outputs critic scores
      output shape is a [batch_size, batch_size] matrix
    baseline_fn (optional): callable that takes y as input 
      outputs a [batch_size]  or [batch_size, 1] vector
    alpha_logit (optional): logit(alpha) for interpolated bound

  Returns:
    scalar estimate of mutual information
  """
  scores = critic_fn(x, y)
  if baseline_fn is not None:
    # Some baselines' output is (batch_size, 1) which we remove here.

    log_baseline = torch.squeeze(baseline_fn(y))
  # if estimator == 'infonce':
  #   mi = infonce_lower_bound_torch(scores)
  if estimator == 'nwj':
    mi = nwj_lower_bound_torch(scores)
  # elif estimator == 'tuba':
  #   mi = tuba_lower_bound_torch(scores, log_baseline)
 
  return mi



class ConcatCriticTorch(torch.nn.Module):
    def __init__(self,hidden_dim, layers, activation, **extra_kwargs):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.mlp = nn.Sequential(
          nn.Linear(2*128,hidden_dim),
          nn.ReLU(),
          nn.Linear(hidden_dim,1),
          # nn.ReLU(),
          # nn.Linear(hidden_dim,1)
        )

        

    def forward(self, x,y):
        
        batch_size = x.size()[0]
        x_t_tiled = torch.tile(x.unsqueeze(dim=0),(batch_size,1,1))
        y_t_tiled = torch.tile(y.unsqueeze(dim=1),(1,batch_size,1))
        xy_pairs = torch.reshape(torch.cat((x_t_tiled, y_t_tiled), dim=2), (batch_size * batch_size, -1))
        scores = self.mlp(xy_pairs)
        return torch.t(torch.reshape(scores, (batch_size, batch_size)))



CRITICS_TORCH = {
    
    'concat': ConcatCriticTorch,
    
}





def mlp(hidden_dim, output_dim, layers, activation):
  return tf.keras.Sequential(
      [tfkl.Dense(hidden_dim, activation) for _ in range(layers)] +
      [tfkl.Dense(output_dim)])
   

class SeparableCritic(tf.keras.Model):
  def __init__(self, hidden_dim, embed_dim, layers, activation, **extra_kwargs):
    super(SeparableCritic, self).__init__()
    self._g = mlp(hidden_dim, embed_dim, layers, activation)
    self._h = mlp(hidden_dim, embed_dim, layers, activation)

  def call(self, x, y):
    scores = tf.matmul(self._h(y), self._g(x), transpose_b=True)
    return scores


class ConcatCritic(tf.keras.Model):
  def __init__(self, hidden_dim, layers, activation, **extra_kwargs):
    super(ConcatCritic, self).__init__()
    # output is scalar score
    self._f = mlp(hidden_dim, 1, layers, activation)

  def call(self, x, y):
    batch_size = tf.shape(x)[0]
    # Tile all possible combinations of x and y
    x_tiled = tf.tile(x[None, :],  (batch_size, 1, 1))
    y_tiled = tf.tile(y[:, None],  (1, batch_size, 1))
    # xy is [batch_size * batch_size, x_dim + y_dim]
    xy_pairs = tf.reshape(tf.concat((x_tiled, y_tiled), axis=2), [batch_size * batch_size, -1])
    # Compute scores for each x_i, y_j pair.
    scores = self._f(xy_pairs) 
    return tf.transpose(tf.reshape(scores, [batch_size, batch_size]))


def gaussian_log_prob_pairs(dists, x):
  """Compute log probability for all pairs of distributions and samples."""
  mu, sigma = dists.mean(), dists.stddev()
  sigma2 = sigma**2
  normalizer_term = tf.reduce_sum(-0.5 * (np.log(2. * np.pi) + 2.0 *  tf.math.log(sigma)), axis=1)[None, :]
  x2_term = -tf.matmul(x**2, 1.0 / (2 * sigma2), transpose_b=True)
  mu2_term = - tf.reduce_sum(mu**2 / (2 * sigma2), axis=1)[None, :]
  cross_term = tf.matmul(x, mu / sigma2, transpose_b=True)
  log_prob = normalizer_term + x2_term + mu2_term + cross_term
  return log_prob

  
def build_log_prob_conditional(rho, **extra_kwargs):
  """True conditional distribution."""
  def log_prob_conditional(x, y):
    mu = x * rho
    q_y = tfd.MultivariateNormalDiag(mu, tf.ones_like(mu) * tf.cast(tf.sqrt(1.0 - rho**2), tf.float32))
    return gaussian_log_prob_pairs(q_y, y)
  return log_prob_conditional
  

CRITICS = {
    'separable': SeparableCritic,
    'concat': ConcatCritic,
    'conditional': build_log_prob_conditional,
}


def log_prob_gaussian(x):
  return tf.reduce_sum(tfd.Normal(0., 1.).log_prob(x), -1)

BASELINES= {
    'constant': lambda: None,
    'unnormalized': lambda: mlp(hidden_dim=512, output_dim=1, layers=2, activation='relu'),
    'gaussian': lambda: log_prob_gaussian,
}


def train_estimator(critic_params, data_params, mi_params, x,y):
  """Main training loop that estimates time-varying MI."""
  # Ground truth rho is only used by conditional critic
  critic = CRITICS[mi_params.get('critic', 'concat')](rho=None, **critic_params)
  baseline = BASELINES[mi_params.get('baseline', 'constant')]()
  
  opt = tf.keras.optimizers.Adam(opt_params['learning_rate'])
  # x,y = x,y
  
  @tf.function
  def train_step(data_params, mi_params,x_,y_):
    # Annoying special case:
    # For the true conditional, the critic depends on the true correlation rho,
    # so we rebuild the critic at each iteration.
    # if mi_params['critic'] == 'conditional':
    #   critic_ = CRITICS['conditional'](rho=rho)
    # else:
    critic_ = critic
    
    with tf.GradientTape() as tape:
      # x, y = sample_correlated_gaussian(dim=data_params['dim'], rho=rho, batch_size=data_params['batch_size'])
      mi = estimate_mutual_information(mi_params['estimator'], x_, y_, critic_, baseline, mi_params.get('alpha_logit', None))
      loss = -mi
  
      trainable_vars = []
      if isinstance(critic, tf.keras.Model):
        trainable_vars += critic.trainable_variables 
      if isinstance(baseline, tf.keras.Model):
        trainable_vars += baseline.trainable_variables
      grads = tape.gradient(loss, trainable_vars)
      opt.apply_gradients(zip(grads, trainable_vars))
    return mi
  
  # Schedule of correlation over iterations 
  # mis = mi_schedule(opt_params['iterations'])
  # rhos = mi_to_rho(data_params['dim'], mis)
  
  estimates = []
  ll = x.size()[1]
  for i in range(ll):
    estimates.append(train_step(data_params, mi_params,x[:,i,:].detach().cpu().numpy(),y[:,i,:].detach().cpu().numpy()).numpy())

  return np.array(estimates)



data_params = {
    'dim': 20,
    'batch_size': 8,
}

critic_params = {
    'layers': 2,
    'embed_dim': 32,
    'hidden_dim': 256,
    'activation': 'relu',
}

opt_params = {
    'iterations': 20000,
    'learning_rate': 5e-4,
}


critic_type = 'concat' # or 'separable'


relu = nn.ReLU()

def train_estimator_torch(critic_params, data_params, mi_params,x,y):
  """Main training loop that estimates time-varying MI."""
  # Ground truth rho is only used by conditional critic
  critic = ConcatCriticTorch(rho=None, **critic_params).to('cuda')
  baseline = None
  opt = torch.optim.Adam(critic.parameters(),lr=opt_params['learning_rate'])
  
  
  
  def train_step(data_params, mi_params,x,y):
    # Annoying special case:
    # For the true conditional, the critic depends on the true correlation rho,
    # so we rebuild the critic at each iteration.
    
    # print(x.size(),y.size())

    # critic_ = critic
    critic.train()
    # for i in critic_.parameters():
    #   i.requires_grad=True
    
    # x, y = sample_correlated_gaussian(dim=data_params['dim'], rho=rho, batch_size=data_params['batch_size'])
    # print(x,y)
    mi = estimate_mutual_information_torch(mi_params['estimator'], x, y, critic, baseline, mi_params.get('alpha_logit', None))
    loss = -mi
    try:
      loss.backward()
    except:
      print('WARNING: in inference mode.....do not call backward')
    opt.step()
    opt.zero_grad()
    critic.eval()
    # for i in critic_.parameters():
    #   i.requires_grad=False
    with torch.no_grad():

      w1 = critic.mlp[0].weight.data.detach().clone()
      w2 = critic.mlp[2].weight.data.detach().clone()
      b1 = critic.mlp[0].bias.data.detach().clone()
      b2 = critic.mlp[2].bias.data.detach().clone()

    
    return w1,w2,b1,b2
  
  
  # # Schedule of correlation over iterations 
  # mis = mi_schedule(opt_params['iterations'])
  # print(mis)
  # rhos = mi_to_rho(data_params['dim'], mis)
  
  # estimates = []
  ll = x.size()[1]
  # print(x.size())
  for i in range(ll):
    w1,w2,b1,b2 = train_step(data_params, mi_params,x[:,i,:].detach().clone(),y[:,i,:].detach().clone())

  return w1,w2,b1,b2
  





estimators = {
    'NWJ': dict(estimator='nwj', critic=critic_type, baseline='constant'),
    # 'TUBA': dict(estimator='tuba', critic=critic_type, baseline='unnormalized'),
    # 'InfoNCE': dict(estimator='infonce', critic=critic_type, baseline='constant'),
    # 'JS': dict(estimator='js', critic=critic_type, baseline='constant'),
    # 'TNCE': dict(estimator='infonce', critic='conditional', baseline='constant'),
    # Optimal critic for TUBA
    #'TUBA_opt': dict(estimator='tuba', critic='conditional', baseline='gaussian')
}

# Add interpolated bounds
def sigmoid(x):
  return 1/(1. + np.exp(-x))
# for alpha_logit in [-5., 0., 5.]:
#   name = 'alpha=%.2f' % sigmoid(alpha_logit)
#   estimators[name] = dict(estimator='interpolated', critic=critic_type,
#                           alpha_logit=alpha_logit, baseline='unnormalized')





device = 'cuda'
import en_core_web_sm, de_core_news_sm
spacy_en = en_core_web_sm.load()






def tokenize_de(text,max_length=20):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)][:max_length-2]

def tokenize_en(text,max_length=20):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)][:max_length-2]


# train = pd.read_csv('./RedditHumorDetection/data/short_jokes/train.tsv', sep='\t', header=None)
# dev = pd.read_csv('./RedditHumorDetection/data/short_jokes/dev.tsv', sep='\t', header=None)

# dd = []
# for i in train.iterrows():
#   ll = {}
#   k = list(i[1])[0].split(',')
#   # print(k)
#   # sents.append(k[3])
#   ll['src'] = k[3]
#   ll['tgt'] = k[3]
#   ll['lab'] = int(k[1])
#   dd.append(ll)
#   # labs.append(int(k[1]))

# # print('len is ***************************************************************** {}'.format(len(dd)))
# # exit()
# # dd = dd[:80]
# dd = dd[:320000]

# train_data_roberta = []
# for i in dd:
#     train_data_roberta.append([i['src'], i['lab']])

# train_df = pd.DataFrame(train_data_roberta)
# train_df.columns = ["text", "labels"]


# dd_t = []
# for i in dev.iterrows():
#   ll = {}
#   k = list(i[1])[0].split(',')
#   # print(k)
#   # sents.append(k[3])
#   ll['src'] = k[3]
#   ll['tgt'] = k[3]
#   ll['lab'] = int(k[1])
#   dd_t.append(ll)
#   # labs.append(int(k[1])
# # print('len is ***************************************************************** {}'.format(len(dd_t)))
# # dd_t = dd_t[:48000]
# dd_t = dd_t[:80]

dir_path = './Sentiment-and-Style-Transfer/data/yelp/'
with open(dir_path+'./sentiment.train.0', 'r') as f:
  k_0 = f.readlines()
k_0 = list(map(lambda x: x.strip(), k_0))

with open(dir_path+'./sentiment.train.1', 'r') as f:
  k_1 = f.readlines()
k_1 = list(map(lambda x: x.strip(), k_1))

train_src = k_0+k_1
train_lab = [0]*len(k_0)+[1]*len(k_1)
train_tgt = train_src

with open(dir_path+'./sentiment.test.0', 'r') as f:
  k_0 = f.readlines()
k_0 = list(map(lambda x: x.strip(), k_0))

with open(dir_path+'./sentiment.test.1', 'r') as f:
  k_1 = f.readlines()
k_1 = list(map(lambda x: x.strip(), k_1))

test_src = k_0+k_1
test_lab = [0]*len(k_0)+[1]*len(k_1)
test_tgt = test_src

dd = [{'src': train_src[i], 'tgt': train_tgt[i], 'lab': train_lab[i]} for i in range(len(train_src))][0:443232]
dd_t = [{'src': test_src[i], 'tgt': test_tgt[i], 'lab': test_lab[i]} for i in range(len(test_src))][0:960]

print(dd)
print('********')
print(dd_t)
print(len(dd))
print(len(dd_t))

# exit(0)
# test_data_roberta = []
# for i in dd_t:
#     test_data_roberta.append([i['src'], i['lab']])

# eval_df = pd.DataFrame(test_data_roberta)
# eval_df.columns = ["text", "labels"]
# from simpletransformers.classification import ClassificationModel, ClassificationArgs
# import pandas as pd
# import logging


# logging.basicConfig(level=logging.INFO)
# transformers_logger = logging.getLogger("transformers")
# transformers_logger.setLevel(logging.WARNING)


# model_args = ClassificationArgs(num_train_epochs=2)
# print(model_args)
# # Create a ClassificationModel
# # roberta = ClassificationModel(
# #     "roberta", "roberta-base", args=model_args
# # )

# # Train the model
# # roberta.train_model(train_df)

# roberta = ClassificationModel(
#     "roberta", "outputs/checkpoint-20000"
# )

# for i in dd_t:
#   predictions, raw_outputs = roberta.predict([i['src']])
#   print(predictions,i['lab'])

# Evaluate the model
# result, model_outputs, wrong_predictions = roberta.eval_model(eval_df)
# print('Result {}'.format(result))


from transformers import DistilBertTokenizerFast,DistilBertForSequenceClassification
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
distilbert = DistilBertForSequenceClassification.from_pretrained('./results_yelp')


# for i in dd_t:
#   inputs = tokenizer(i['src'], return_tensors="pt")

#   with torch.no_grad():
#       logits = model(**inputs).logits

#   predicted_class_id = logits.argmax().item()
#   print(predicted_class_id,i['lab'])


# arguments for Trainer
# test_args = TrainingArguments(
#     output_dir = './results_infer',
#     do_train = False,
#     do_predict = True,
#     per_device_eval_batch_size = 8,   
#     # dataloader_drop_last = False    
# )

# # init trainer
# trainer = Trainer(
#               model = model, 
#               args = test_args, 
#               # compute_metrics = compute_metrics
#               )

# test_results = trainer.predict(test_dataset)
# print(test_results)

# preds = np.argmax(test_results.predictions,axis=-1)

import json
with open("./data_yelp.json", 'w') as f:
    for item in dd:
        f.write(json.dumps(item) + "\n")

import json
with open("./data_dev_yelp.json", 'w') as f:
    for item in dd_t:
        f.write(json.dumps(item) + "\n")

from torchtext.legacy import data
from torchtext.legacy import datasets

LAB = data.Field(sequential=False, use_vocab=False)


SRC = Field(tokenize =tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            # fix_length=100,
            batch_first = True)

TRG = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            # fix_length=100,
            batch_first = True)

fields = {'src': ('s', SRC), 'tgt': ('t', TRG), 'lab': ('l', LAB)}

train_data,test_data  = data.TabularDataset.splits(
                            path = '.',
                            train = 'data_yelp.json',
                            test = 'data_dev_yelp.json',
                            format = 'json',
                            fields = fields
)

SRC.build_vocab(train_data, min_freq = 1)
TRG.build_vocab(train_data, min_freq = 1)

BATCH_SIZE = 32

train_iterator,test_iterator = BucketIterator.splits(
    (train_data,test_data), 
    sort_key = lambda x: len(x.s),
     batch_size = BATCH_SIZE,
     device = device)


print(len(train_iterator))

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention



class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = 20):
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.ffn = nn.Linear(hid_dim,2)
        
    def forward(self, src, src_mask, is_discriminator=False, if_req=False):
        
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        # print(torch.max(src),torch.min(src))
        # print(src_mask.shape)
        # print('tok embedding shape {}'.format(self.tok_embedding.weight.repeat((8,1,1)).shape))
        batch_size = src_mask.shape[0]
        src_len = src_mask.shape[3]
        # src_len = min(src_len,100)
        # src = src[:,0:src_len]
        # src_mask = src_mask[:,:,:,0:src_len]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, src len]
        
        if if_req:
          # print('in if_req')
          # print(src.shape)
          # print(self.tok_embedding.weight.repeat((8,1,1)).shape)
          
          src_ = self.dropout( (torch.bmm(src,self.tok_embedding.weight.repeat((32,1,1)))*self.scale) + self.pos_embedding(pos) )
        else:
          src_ = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        
        #src = [batch size, src len, hid dim]
        
        src__ = self.layers[0](src_,src_mask)
        for layer in self.layers[1:]:
            src__ = layer(src__, src_mask)
            
        #src = [batch size, src len, hid dim]
        
        if if_req:
          return src__,src_


        if not is_discriminator:
          return src__
        else:
          return self.ffn(src__.mean(dim=1))


class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
        # print(torch.max(src))
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        return torch.tensor(inp.long(),requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        print(F.hardtanh(grad_output))
        return F.hardtanh(grad_output)
class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = 20):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.max_length = max_length
        self.ste = StraightThroughEstimator()
        
    def forward(self, trg, enc_src, trg_mask, src_mask, is_policy=False, gumbel_softmax=False):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        # print(trg_len)
        # trg_len = min(trg_len,100)
        # trg = trg[:,0:trg_len]
        # trg_mask= trg_mask[:,:,0:trg_len,0:trg_len]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        #pos = [batch size, trg len]
            
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
                
        #trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        output = self.fc_out(trg)

        if is_policy:
            # pred_token = output.argmax(2)[:,-1].item()
            # output_shape = (bs,length,hidden_dim)
            output = output[:,-1,:] #select output token from last state
            token_probs = F.softmax(output,dim=1) # softmax prob distribution
            m = Categorical(token_probs)
            action = m.sample()
            
            log_prob = m.log_prob(action)
            return action, log_prob
        # https://www.hassanaskary.com/python/pytorch/deep%20learning/2020/09/19/intuitive-explanation-of-straight-through-estimators.html
        if gumbel_softmax:
          gumbel_one_hot = F.gumbel_softmax(output[:,-1,:].squeeze(),tau=1,hard=True)
          # print(gumbel_one_hot)
          # print(gumbel_one_hot.size())
          ll = int(list(gumbel_one_hot.size())[1])
          aranged = torch.arange(ll).repeat(32,1).to('cuda')
          # print(aranged,aranged.shape)

          hard_sample_w_grad = torch.sum(aranged*gumbel_one_hot,axis=1) # reparameterization to get one hot vector
          hard_sample_w_grad = hard_sample_w_grad.unsqueeze(0)
          # hard_sample_wo_grad = self.ste(hard_sample_w_grad)
          hard_sample_wo_grad = hard_sample_w_grad.long()

          return output, attention, hard_sample_wo_grad, gumbel_one_hot



        
        #output = [batch size, trg len, output dim]
            
        return output, attention


class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
            
        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        #trg = [batch size, trg len, hid dim]
        
        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return trg, attention



INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 128
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 256
DEC_PF_DIM = 256
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
print(INPUT_DIM)
class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder_humor,
                 decoder_nonhumor, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder_humor = decoder_humor
        self.decoder_nonhumor = decoder_nonhumor
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

        self.style_proj_h = nn.Linear(HID_DIM,HID_DIM)
        self.style_proj_nh = nn.Linear(HID_DIM,HID_DIM)
        self.content_proj = nn.Linear(HID_DIM,HID_DIM)
        

        self.style_labeler = nn.Linear(HID_DIM,2)
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)

        #divide enc_src to two separate vectors having same dimension as enc_src but should encode style and contents respectively

        style_h = self.style_proj_h(enc_src)
        style_nh = self.style_proj_nh(enc_src)

        # style_h_ = torch.tensor(style_h.detach().cpu().numpy()).to('cuda')
        # style_nh_ = torch.tensor(style_nh.detach().cpu().numpy()).to('cuda')

        # for estimator, mi_params in estimators.items():
        #     # print("Training %s..." % estimator)
        #     w1,w2,b1,b2 = train_estimator_torch(critic_params, data_params, mi_params, style_h_, style_nh_)
        #  # = train_estimator_torch(critic_params, data_params, mi_params, style, content)



        # w1 = torch.tensor(w1.detach().cpu().numpy()).to('cuda')
        # w2 = torch.tensor(w2.detach().cpu().numpy()).to('cuda')
        # b1 = torch.tensor(b1.detach().cpu().numpy()).to('cuda')
        # b2 = torch.tensor(b2.detach().cpu().numpy()).to('cuda')

        # estimates = []
        # # print(w1,w2,b1,b2)
        # x,y = style_h,style_nh
        # ll = x.size()[1]
        # for i in range(ll):


        #   # w1,w2,b1,b2 = train_step(data_params, mi_params,x[:,i,:].detach().clone(),y[:,i,:].detach().clone())
          
        #   batch_size = x[:,i,:].size()[0]
        #   x_t_tiled = torch.tile(x[:,i,:].unsqueeze(dim=0),(batch_size,1,1))
        #   y_t_tiled = torch.tile(y[:,i,:].unsqueeze(dim=1),(1,batch_size,1))
        #   xy_pairs = torch.reshape(torch.cat((x_t_tiled, y_t_tiled), dim=2), (batch_size * batch_size, -1))
        #   # print(xy_pairs.size(),w1.size())
        #   k = relu(xy_pairs.mm(w1)+b1)
        #   # print(k.size(),w2.size())
        #   scores = k.mm(torch.t(w2))+b2
          
        #   k_ = torch.t(torch.reshape(scores, (batch_size, batch_size)))
        #   # print(k_)
        #   # print(k_.size())
        #   mi_updated = nwj_lower_bound_torch(k_)
        #   # mi_updated.backward()
        #   # print(mi_updated)
        #   # print(mi_updated.size())
        #   estimates.append(mi_updated)

        # estimates = torch.stack(estimates)

        # mi_alpha = torch.mean(estimates)

        pred_style = self.style_labeler(torch.mean(style_h,dim=1))

        content = self.content_proj(enc_src)

        #style and content's resultant should reconstruct input sentence,so,
        enc_out_h = style_h+content
        enc_out_nh = style_nh+content
        
        #enc_src = [batch size, src len, hid dim]

        
                
        output_h, attention_h = self.decoder_humor(trg, enc_out_h, trg_mask, src_mask)
        output_n, attention_n = self.decoder_nonhumor(trg, enc_out_nh, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        mi_alpha=0.0 #dummy
        return output_h, attention_h, output_n,attention_n, style_h,style_nh, content, pred_style, mi_alpha




enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)

disc_nh = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)

disc_h = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)


dec_h = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)


dec_n = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)




SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = Seq2Seq(enc, dec_h,dec_n, SRC_PAD_IDX, TRG_PAD_IDX, device)
state_dict = torch.load('./double_decoder_rl_cc_yelp/tr_model_full_reward_epoch4.pt')
# print(state_dict)
from collections import OrderedDict
state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
# print(state_dict)
model.load_state_dict(state_dict)
optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)
import argparse

import deepspeed
def add_argument():

   parser=argparse.ArgumentParser(description='CIFAR')

   #data
   # cuda
   parser.add_argument('--with_cuda', default=False, action='store_true',
                       help='use CPU in case there\'s no GPU support')
   parser.add_argument('--use_ema', default=False, action='store_true',
                       help='whether use exponential moving average')

   # train
   parser.add_argument('-b', '--batch_size', default=32, type=int,
                       help='mini-batch size (default: 32)')
   parser.add_argument('-e', '--epochs', default=30, type=int,
                       help='number of total epochs (default: 30)')
   parser.add_argument('--local_rank', type=int, default=-1,
                      help='local rank passed from distributed launcher')
  

   # Include DeepSpeed configuration arguments
   parser = deepspeed.add_config_arguments(parser)

   args=parser.parse_args()

   return args

args=add_argument()
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

# model.apply(initialize_weights);

LEARNING_RATE = 0.0005



criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX,  reduction='none')
criterion_style = nn.CrossEntropyLoss()


print("DeepSpeed is enabled.")

model, optimizer, _, _ = deepspeed.initialize(
 
  model=model,
  optimizer=optimizer,
  config_params='./ds_config.json'
)




def flip(p):
    return 'H' if random.random() < p else 'T'
from evaluate import load
# bleurt = load("bleurt", module_type="metric")
def get_reward(a,b,lab, j= False):
    # if j:
        
    #     a = a
    # else:
    #     a = ' '.join(a)

    # b = ' '.join(b)

    p = []
    for i in a.split(' '):
      if i== '<eos>':
        break
      else:
        p.append(i)

    a = ' '.join(p)

    # ble = bleurt.compute(predictions=[a], references=[b])['scores'][0]
    ble = (sentence_bleu(a,[b]).score)/100
    # predictions, raw_outputs = roberta.predict([a])
    # pred_style = predictions[0]
    inputs = tokenizer(a, return_tensors="pt")

    with torch.no_grad():
        logits = distilbert(**inputs,return_dict=True).logits

    pred_style = logits.argmax().item()
    # print(pred_style)
    # rew = 0
    # if ble>0:
    #     rew+=2
    # elif ble<0 and ble>-0.5:
    #     rew+=1
    # elif ble<-0.5:
    #     rew+=0

    # if pred_style==1:
    #     rew+=5
    # else:
    #     rew+=0

    # return rew

    if j:
        print('Greedy : {}'.format(a))
    else:
        print('Sampled : {}'.format(a))

    print('Reference : {}'.format(b))
    print('BLEURT: {}'.format(ble))
    print('pred style: {}'.format(pred_style))
    print('label : {}'.format(lab))
    if lab==0:
        print('training humorous decoder via RL')
    elif lab==1:
        print('training non-humorous decoder via RL')
    print('**************************************')
    if lab==0:
      #trainining humorous decoder only
      return (ble+pred_style)/2
    elif lab==1:
      #training non humorous decoder only
      if pred_style==0:
        return (1+ble)/2
      else:
        return ble/2
    # if lab==0:
    #     #training humorous decoder via RL
    #     return 5*pred_style+4*ble
    # elif lab==1:
    #     # training non humorous decoder via RL
    #     if pred_style==0:
    #         return 5+4*ble
    #     else:
    #         return 4*ble
    # return 2*max(0,ble)+5*pred_style
    # return 2*ble+5*pred_style




from tqdm import tqdm
REWARDS = [0]
CCC = [0]
TURN=0
def reinforce(list_of_src_tokens,list_of_trg_tokens, src_field, trg_field, model, device,lab_, max_len = 18):
    

    n_training_episodes=32
    # optimizer = optim.Adam(model.parameters(), lr=0.0005)

    scores_deque = deque(maxlen=10)
    scores = []

    losses = []

    model.train()
    tot_loss=0  
    lot = []
    for i_episode in tqdm(range(n_training_episodes)):
      s = []
      for i in list_of_trg_tokens[i_episode,:]:
        if TRG.vocab.itos[i.item()]=='<pad>':
          break
        s.append(TRG.vocab.itos[i.item()])

           
            
      trg_sentence = ' '.join(s[1:-1])
      lot.append(trg_sentence)
    # print(lot)

    # print(list_of_src_tokens)
    src_tensor = list_of_src_tokens
    src_mask = model.make_src_mask(src_tensor)
    
    # with torch.no_grad():
    enc_src = model.encoder(src_tensor, src_mask)
    style_h = model.style_proj_h(enc_src)
    style_nh = model.style_proj_nh(enc_src)

    pred_style = model.style_labeler(torch.mean(style_h,dim=1))

    content = model.content_proj(enc_src)

    #style and content's resultant should reconstruct input sentence,so,
    # enc_out = style+content

   
    enc_out_h = style_h+content
    enc_out_nh = style_nh+content
    # print('enc out h {}'.format(enc_out_h.size()))
    trg_indexes_h,trg_indexes_nh = [],[]

    trg_indexes_h.append(np.asarray([trg_field.vocab.stoi[trg_field.init_token]]*32))
    trg_indexes_nh.append(np.asarray([trg_field.vocab.stoi[trg_field.init_token]]*32))
    pred_toks_h = []
    pred_toks_nh = []
    W_h,W_nh = [],[]
    for i in range(max_len):
        
        trg_tensor_h = torch.LongTensor(trg_indexes_h).T.to(device)
        trg_tensor_nh = torch.LongTensor(trg_indexes_nh).T.to(device)
        # print(trg_tensor_h,trg_tensor_h.shape)
        # trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask_h = model.make_trg_mask(trg_tensor_h)

        
        # trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask_nh = model.make_trg_mask(trg_tensor_nh)
        
        # print(trg_tensor,trg_mask)
        # exit(0)
        # with torch.no_grad():
        output_h, attention_h, sample_h, w_grad_h = model.decoder_humor(trg_tensor_h, enc_out_h, trg_mask_h, src_mask,gumbel_softmax=True)
        output_nh, attention_nh, sample_nh, w_grad_nh = model.decoder_nonhumor(trg_tensor_nh, enc_out_nh, trg_mask_nh, src_mask,gumbel_softmax=True)
        # print(output_h.shape,w_grad_h.shape)
        # print()
        # print('**********')
        # print(output_nh.shape,w_grad_nh.shape)
        
        
        
    
        # w_grad = w_grad.unsqueeze(0)

        pred_token_h = output_h.argmax(2)[:,-1].detach().cpu().numpy()
        pred_token_nh = output_nh.argmax(2)[:,-1].detach().cpu().numpy()
        # print(pred_token)
        # exit(0)
        # print(sample)
        # print(hard_sample_w_grad,pred_token)
        # pred_toks.append(hard_sample_w_grad.unsqueeze(0))
        # pred_toks.append(sample)
        W_h.append(w_grad_h)
        W_nh.append(w_grad_nh)
        pred_toks_h.append(sample_h)
        pred_toks_nh.append(sample_nh)
        # print(sample,w_grad)
        trg_indexes_h.append(pred_token_h)
        trg_indexes_nh.append(pred_token_nh)
        # print(trg_indexes_h)

        
    
    
    # print('((()()()(()()()()()()()()()())))))')
    # trg_tokens_h = [trg_field.vocab.itos[i] for i in trg_indexes_h]
    # trg_tokens_nh = [trg_field.vocab.itos[i] for i in trg_indexes_nh]

    # pred_toks = torch.cat(pred_toks).unsqueeze(0)

    W_h = torch.stack(W_h)
    W_nh = torch.stack(W_nh)
    trg_indexes_h= np.asarray(trg_indexes_h)[1:,:]
    trg_indexes_nh= np.asarray(trg_indexes_nh)[1:,:]
    # print(np.shape(trg_indexes_h))
    # print(np.shape(trg_indexes_nh))
    # print(W_h.shape)
    # print(W_nh.shape)

    W = []
    pred_toks = []
    target = []
    

    # print(pred_toks)
    # pred_toks = pred_toks.type(torch.LongTensor)
    # print(pred_toks)
    # return trg_tokens[1:], attention, pred_toks, W



    pred_toks_h = torch.cat(pred_toks_h)
    pred_toks_nh = torch.cat(pred_toks_nh)
    # print(pred_toks_h.shape,pred_toks_nh.shape)
    # W_h = torch.cat(W_h)
    # W_nh = torch.cat(W_nh)

    # print(W_h.shape,W_nh.shape)
    for i_,i in enumerate(lab_):
      # print(i.item())
      if i.item()==0:
        W.append(W_h[:,i_,:])
        pred_toks.append(pred_toks_h[:,i_])
        target.append(trg_indexes_h[:,i_])
      else:
        W.append(W_nh[:,i_,:])
        pred_toks.append(pred_toks_nh[:,i_])
        target.append(trg_indexes_nh[:,i_])
    
    W = torch.stack(W)
    pred_toks = torch.stack(pred_toks)
    # target = torch.LongTensor(target)

    # print(W.shape,pred_toks.shape,np.shape(target))

    # baseline  []
    baseline = [[trg_field.vocab.itos[i] for i in trg] for trg in target]

    baseline = list(map(lambda i: ' '.join(i[:]),baseline))

    # print(baseline)
    bs_toks_soft = W
    bs_toks = pred_toks



    # print('*****************************************************************************************************')
    enc_src = model.encoder(src_tensor, src_mask)
    style_h = model.style_proj_h(enc_src)
    style_nh = model.style_proj_nh(enc_src)

    pred_style = model.style_labeler(torch.mean(style_h,dim=1))

    content = model.content_proj(enc_src)

    #style and content's resultant should reconstruct input sentence,so,
    # enc_out = style+content

   
    enc_out_h = style_h+content
    enc_out_nh = style_nh+content
    # print('enc out h {}'.format(enc_out_h.size()))
    trg_indexes_h,trg_indexes_nh = [],[]

    trg_indexes_h.append(np.asarray([trg_field.vocab.stoi[trg_field.init_token]]*32))
    trg_indexes_nh.append(np.asarray([trg_field.vocab.stoi[trg_field.init_token]]*32))
    logprob_h = []
    logprob_nh = []
    for i in range(max_len):
        
        trg_tensor_h = torch.LongTensor(trg_indexes_h).T.to(device)
        trg_tensor_nh = torch.LongTensor(trg_indexes_nh).T.to(device)
        # print(trg_tensor_h,trg_tensor_h.shape)
        # trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask_h = model.make_trg_mask(trg_tensor_h)

        
        # trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask_nh = model.make_trg_mask(trg_tensor_nh)
        
        # print(trg_tensor,trg_mask)
        # exit(0)
        # with torch.no_grad():
        # output_h, attention_h, sample_h, w_grad_h = model.decoder_humor(trg_tensor_h, enc_out_h, trg_mask_h, src_mask,gumbel_softmax=True)
        action_h, log_prob_h = model.decoder_humor(trg_tensor_h, enc_out_h, trg_mask_h, src_mask, is_policy=True)
        action_nh, log_prob_nh = model.decoder_nonhumor(trg_tensor_nh, enc_out_nh, trg_mask_nh, src_mask, is_policy=True)
        trg_indexes_h.append(action_h.detach().cpu().numpy())
        trg_indexes_nh.append(action_nh.detach().cpu().numpy())
        logprob_h.append(log_prob_h)
        logprob_nh.append(log_prob_nh)

    logprob_h = torch.stack(logprob_h)
    logprob_nh = torch.stack(logprob_nh)
    trg_indexes_h = np.asarray(trg_indexes_h)[1:,:]
    trg_indexes_nh = np.asarray(trg_indexes_nh)[1:,:]
    # print(logprob_h.shape,np.shape(trg_indexes_h))

    logprob = []
    action = []
    for i_,i in enumerate(lab_):
      # print(i.item())
      if i.item()==0:

        logprob.append(logprob_h[:,i_])
        action.append(trg_indexes_h[:,i_])
      else:
        
        logprob.append(logprob_nh[:,i_])
        action.append(trg_indexes_nh[:,i_])
    
    
    logprob = torch.stack(logprob)
    action = np.asarray(action)
    # print(action)
    action_mask = []
    eos_tok = trg_field.vocab.stoi[trg_field.eos_token]
    for i in action:
      try:
        p = [1]*list(i).index(eos_tok)+[0]*(len(list(i))-list(i).index(eos_tok))
      except:
        p = [1]*len(list(i))
      action_mask.append(p)
    action_mask = np.asarray(action_mask)
    # print(action)
    # print(action_mask)


    print(logprob.shape, np.shape(action_mask))

    final_sen = [[trg_field.vocab.itos[i] for i in trg] for trg in action]

    final_sen = list(map(lambda i: ' '.join(i[:]),final_sen))
    # print(lot, baseline, final_sen)
    # print(len(lot), len(baseline), len(final_sen))
    batch_baseline_reward = []
    batch_sample_reward = []
    for i in range(32):
      baseline_reward = get_reward(baseline[i],lot[i],lab_[i].item(),j=True)
      sample_reward = get_reward(final_sen[i],lot[i],lab_[i].item(),j=False)
      batch_baseline_reward.append(baseline_reward)
      batch_sample_reward.append(sample_reward)
    # print(batch_sample_reward)
    # print(batch_baseline_reward)

    R = np.tile(np.asarray(batch_baseline_reward)-np.asarray(batch_sample_reward), (max_len,1)).T

    R = R*action_mask # mask tokens obtained after EOS token by 0 to not assign any reward to them.
    print(R)
    print(np.shape(R),logprob.shape)
    # print(R)
    REWARDS.extend(list(np.asarray(batch_baseline_reward)))

    rl_loss = torch.mean(torch.sum(logprob*torch.tensor(R).to('cuda'),axis=1))
    # print(rl_loss)

    baseline_mask = model.make_src_mask(bs_toks)
    # print(baseline_mask)
    
    # with torch.no_grad():
    enc_baseline, enc_grad = model.encoder(bs_toks_soft, baseline_mask, if_req=True)
    style_h = model.style_proj_h(enc_baseline)

    style_nh = model.style_proj_nh(enc_baseline)

    

    content = model.content_proj(enc_baseline)

    #style and content's resultant should reconstruct input sentence,so,
    # enc_out = style+content

    # print('here')
    
    enc_out_h = style_h+content
    enc_out_nh = style_nh+content

    # print(enc_out_h.shape, enc_out_nh.shape)
    # print(src_tensor)
    trg_tensor=src_tensor[:,:-1]
    trg_mask = model.make_src_mask(trg_tensor)
    # print(trg_tensor,trg_mask)
    
    op_nh,attn_nh = model.decoder_nonhumor(trg_tensor, enc_out_nh, trg_mask, baseline_mask)
    
    op_h,attn_h = model.decoder_humor(trg_tensor, enc_out_h, trg_mask, baseline_mask)

    # print(op_h.shape)

    op_h = op_h.contiguous().view(-1, op_h.shape[-1])
    op_nh = op_nh.contiguous().view(-1, op_nh.shape[-1])

    # print(op_h.shape)

    trg_ = src_tensor[:,1:].contiguous().view(-1)

    # print(op,trg_)
    # trg_ = src_tensor[:,1:]
    # print(trg_.shape)
    cyclic_consistency_loss_h = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX,  reduction='none')(op_h, trg_)
    cyclic_consistency_loss_nh = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX,  reduction='none')(op_nh, trg_)
    # print(cyclic_consistency_loss_h)
    # print(cyclic_consistency_loss_nh)

    l1_s = int(cyclic_consistency_loss_h.size()[0]/32)
    l2_s = int(cyclic_consistency_loss_nh.size()[0]/32)

    cyclic_consistency_loss_h = cyclic_consistency_loss_h.reshape((32,l1_s))
    cyclic_consistency_loss_nh = cyclic_consistency_loss_nh.reshape((32,l2_s))
    l1 = lab_.repeat_interleave(l1_s).reshape(32,-1)
    l2 = lab_.repeat_interleave(l2_s).reshape(32,-1)

   
    loss_ce1 = l1*cyclic_consistency_loss_h
    loss_ce2 = (l2^1)*cyclic_consistency_loss_nh

    cc_loss = (loss_ce1.mean()+loss_ce2.mean())/2
    CCC.append(cc_loss.item())

    print('AVG Reward {} || Avg Cyclic Loss {}'.format(np.mean(np.array(REWARDS)),np.mean(np.array(CCC))))

    tot_loss = rl_loss+cc_loss
    return tot_loss
















      
    # exit(0)
    






    # for i_episode in tqdm(range(n_training_episodes)):
        
    #     lab = lab_[i_episode].item()
    #     saved_log_probs = []
    #     rewards = []
    #     # sentence = list_of_src_sentences[i_episode]
    #     # print(sentence)
    #     # trg_sentence = list_of_trg_sentences[i_episode]
    #     # if isinstance(sentence, str):
    #     #     # nlp = spacy.load('de_core_news_sm')
    #     #     nlp = en_core_web_sm.load()
    #     #     tokens = [token.text.lower() for token in nlp(sentence)]
    #     # else:
    #     #     tokens = [token.lower() for token in sentence]

    #     s = []
    #     for i in list_of_trg_tokens[i_episode,:]:
            
            
    #         if TRG.vocab.itos[i.item()]=='<pad>':
    #             # print('reached')
    #             break
    #         s.append(TRG.vocab.itos[i.item()])

    #         # print(len(s))
            
    #     trg_sentence = ' '.join(s[1:-1])
    #     # print(trg_sentence)

    #     s = []
    #     for i in list_of_src_tokens[i_episode,:]:
            
            
    #         if SRC.vocab.itos[i.item()]=='<eos>':
    #             # print('reached')
    #             break
    #         s.append(SRC.vocab.itos[i.item()])

    #         # print(len(s))
            
    #     sentence = s[1:]
    #     sentence = ' '.join(sentence)

    #     baseline,_,bs_toks,bs_toks_soft = translate_sentence_for_baseline(sentence, SRC, TRG, model, device,lab, max_len = 50)
    #     # print('bs_toks {}'.format(bs_toks))
    #     bs_toks_soft = bs_toks_soft.squeeze(0)
    #     # print('bs_toks soft {}'.format(bs_toks_soft))
    #     baseline = ' '.join(baseline[:-1])
    #     # print(sentence)

    #     # print('************')
        
    #     if isinstance(sentence, str):
    #         # nlp = spacy.load('de_core_news_sm')
    #         nlp = en_core_web_sm.load()
    #         tokens = [token.text.lower() for token in nlp(sentence)]
    #     else:
    #         tokens = [token.lower() for token in sentence]


    #     # if len(tokens)>=98:
    #     #     print('here')
    #     #     tokens=tokens[0:97]
    #     tokens = tokens[0:98]

    #     # tokens = list(list_of_src_tokens[i_episode,:].detach().cpu().numpy())

    #     # print(tokens)
    #     tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    #     remaining = 100-len(tokens)

    #     tokens = tokens+[src_field.pad_token]*remaining

    #     # print(tokens)
            
    #     src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    #     # print(src_indexes)

    #     src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    #     # print('src_tensor shape {}'.format(src_tensor.size()))
        
    #     src_mask = model.make_src_mask(src_tensor)
        
    #     # with torch.no_grad():
    #     # print('src tensor, src mask')
    #     # print(src_tensor, src_mask)
    #     enc_src = model.encoder(src_tensor, src_mask)
    #     style_h = model.style_proj_h(enc_src)

    #     style_nh = model.style_proj_nh(enc_src)

    #     pred_style = model.style_labeler(torch.mean(style_h,dim=1))

    #     content = model.content_proj(enc_src)

    #     #style and content's resultant should reconstruct input sentence,so,
    #     # enc_out = style+content


        
    #     enc_out_h = style_h+content
    #     enc_out_nh = style_nh+content

    #     trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    #     full_sen = []

    #     for i in range(max_len):

    #         trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

    #         trg_mask = model.make_trg_mask(trg_tensor)
            
    #         # with torch.no_grad():
    #         if lab==0:
    #             #lab non humorous, use RL for decoder_h
    #             # print('label non humorous : RL training for decoder_humor')
    #             action, log_prob = model.decoder_humor(trg_tensor, enc_out_h, trg_mask, src_mask, is_policy=True)
    #         if lab==1:
    #             # print('label humorous : RL training for decoder_nonhumor')
    #             action, log_prob = model.decoder_nonhumor(trg_tensor, enc_out_nh, trg_mask, src_mask, is_policy=True)


            
            
    #         full_sen.append(trg_field.vocab.itos[action.item()])
    #         rewards.append(0)
    #         saved_log_probs.append(log_prob)
    #         trg_indexes.append(action.item())
    #         if action.item() == trg_field.vocab.stoi[trg_field.eos_token]:
    #             break
    #     final_reward = get_reward(full_sen[:-1],trg_sentence,lab)
    #     rewards.append(final_reward)
    #     scores_deque.append(sum(rewards))
    #     scores.append(sum(rewards))
    #     reward_baseline=get_reward(baseline,trg_sentence,lab,j=True)

    #     R = reward_baseline-sum(rewards) # no discounting
    #     REWARDS.append(sum(rewards))

    #     # print(R)
    #     # print(saved_log_probs)
    #     policy_loss = []
    #     for log_prob in saved_log_probs:
    #         policy_loss.append(log_prob * R)
    #     # print(policy_loss)
    #     policy_loss = torch.cat(policy_loss).sum()
    #     # print(policy_loss)
    #     tot_loss+=policy_loss
    #     losses.append(policy_loss)
    #     # Line 8:
    #     # optimizer.zero_grad()
    #     # policy_loss.backward()
    #     # optimizer.step()
    #     # if i_episode % 50 == 0:
    #     # print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    #     #baseline is the greedily generated sentence
    #     # tokenized_baseline = [src_field.init_token] + [token.text.lower() for token in nlp(baseline)][0:98] + [src_field.eos_token]
    #     # #input tokenized baseline to the model again
    #     # #if baseline is generated by decoder_humor, then label was 0, then non_humorous decoder should decode source tokens
    #     # source_token = list_of_src_tokens[i_episode,:]
    #     # tokenized_baseline = [src_field.vocab.stoi[token] for token in tokenized_baseline]

    #     # # print(src_indexes)

    #     # tokenized_baseline = torch.LongTensor(tokenized_baseline).unsqueeze(0).to(device)
    #     # print('tokenized_baseline {}; src tokens {}'.format(tokenized_baseline,source_token))
    #     # print(bs_toks)
    #     # print(bs_toks.size())
    #     baseline_mask = model.make_src_mask(bs_toks)
    #     # print(baseline_mask)
        
    #     # with torch.no_grad():
    #     enc_baseline, enc_grad = model.encoder(bs_toks_soft, baseline_mask, if_req=True)
    #     style_h = model.style_proj_h(enc_baseline)
    #     baseline_mask = model.make_src_mask(bs_toks)
    #     # print(baseline_mask)
        
    #     # with torch.no_grad():
    #     enc_baseline, enc_grad = model.encoder(bs_toks_soft, baseline_mask, if_req=True)
    #     style_h = model.style_proj_h(enc_baseline)

    #     style_nh = model.style_proj_nh(enc_baseline)

       

    #     content = model.content_proj(enc_baseline)

    #     #style and content's resultant should reconstruct input sentence,so,
    #     # enc_out = style+content

    #     # print('here')
        
    #     enc_out_h = style_h+content
    #     enc_out_nh = style_nh+content

    #     #style and content's resultant should reconstruct input sentence,so,
    #     # enc_out = style+content

    #     # print('here')
        
    #     enc_out_h = style_h+content
    #     enc_out_nh = style_nh+content
    #     trg_tensor=src_tensor[:,:-1]
    #     trg_mask = model.make_src_mask(trg_tensor)
    #     # print(trg_tensor,trg_mask)
    #     if lab==0:
    #       op,attn = model.decoder_nonhumor(trg_tensor, enc_out_nh, trg_mask, baseline_mask)
    #     if lab==1:
    #       op,attn = model.decoder_humor(trg_tensor, enc_out_h, trg_mask, baseline_mask)

    #     # print(op)

    #     op = op.contiguous().view(-1, op.shape[-1])

    #     # print(op.grad_fn)

    #     trg_ = src_tensor[:,1:].contiguous().view(-1)

    #     # print(op,trg_)
    #     cyclic_consistency_loss = criterion(op, trg_).mean()
    #     CCC.append(cyclic_consistency_loss.item())

    #     # cyclic_consistency_loss.backward(retain_graph=True)

    #     # print(bs_toks.grad,enc_baseline.grad, op.grad)
    #     # print(enc_grad.grad, W.grad)
    #     # print(torch.autograd.grad(cyclic_consistency_loss,enc_grad))
    #     # print(torch.autograd.grad(cyclic_consistency_loss,bs_toks_soft))

    #     print(tot_loss,cyclic_consistency_loss)
    #     mean_reward = sum(REWARDS)/len(REWARDS)
    #     mean_ccc = sum(CCC)/len(CCC)
    #     print('running mean reward {}'.format(mean_reward))
    #     print('running mean cyclic_consistency_loss {}'.format(mean_ccc))

    #     tot_loss+=cyclic_consistency_loss






            
    



# for outer_loop in range(5):
#     print('Outer Loop No. {}'.format(outer_loop))

#     N_EPOCHS = 4
#     CLIP = 1

#     best_valid_loss = float('inf')

#     for epoch in range(N_EPOCHS):
        
#         start_time = time.time()
        
#         train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
#         valid_loss = evaluate(model, test_iterator, criterion)
        
#         end_time = time.time()
        
#         epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
#         if valid_loss < best_valid_loss:
#             best_valid_loss = valid_loss
#             torch.save(model.state_dict(), './tr_model.pt')
        
#         print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
#         print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
#         print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

#     neg_humor_vec = []
#     pos_humor_vec = []


#     print('constructing style vectors...')
#     for i in range(len(train_iterator)):
#         style,lab = get_vectors(i,SRC,TRG,model,'cuda')
#         if lab==0:
#             neg_humor_vec.append(style)
#         else:
#             pos_humor_vec.append(style)


#     print(len(neg_humor_vec))
#     print(len(pos_humor_vec))
#     print(neg_humor_vec[0])
#     print(neg_humor_vec[0].size())


#     neg_humor_vec = torch.stack(neg_humor_vec).squeeze(dim=1).mean(dim=0)
#     pos_humor_vec = torch.stack(pos_humor_vec).squeeze(dim=1).mean(dim=0)

#     print(neg_humor_vec.size())
#     print(pos_humor_vec.size())







def train(model, iterator, optimizer, criterion, clip, epoch_nos):
    
    model.train()
    print('here')
    epoch_loss = 0
    
    for i_, batch in tqdm(enumerate(iterator)):
        
        src = batch.s
        trg = batch.t
        lab = batch.l
        # print(lab)
        los,lot = [],[]
        for i in src:
            s = []
            for j in i:
                s.append(SRC.vocab.itos[j.item()])
            # print(len(s))
            s = s[0:20]
            los.append(' '.join(s))

        # for i in trg:
        #     s = []
        #     for j in i:
        #         s.append(TRG.vocab.itos[j.item()])
        #     print(len(s))
        #     s  = s[0:100]
        #     lot.append(' '.join(s))


        
        # print(f'los {los}')
        # print(f'lot {lot}')


        # optimizer.zero_grad()
        # print(src)
        # print(trg)
        # exit(0)
        output_h, _,output_nh,_, style_h,style_nh, content, pred_style,mi = model(src, trg[:,:-1])
                
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
            
        output_dim_h = output_h.shape[-1]
            
        output_h = output_h.contiguous().view(-1, output_dim_h)

        output_dim_nh = output_nh.shape[-1]
            
        output_nh = output_nh.contiguous().view(-1, output_dim_nh)

        trg = trg[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]

        result = torch.einsum('abc, abc->ab', style_h, style_nh)
        bs = style_h.size()[0]
        ll = style_h.size()[1]
        loss_mse = nn.MSELoss()(result, torch.zeros((bs,ll)).to(device))
        print(output_h.shape,trg.shape)
       
        loss_ce1 = criterion(output_h, trg)
        loss_ce2 = criterion(output_nh, trg)
        loss_st = criterion_style(pred_style,lab)
        l1_s = int(loss_ce1.size()[0]/32)
        l2_s = int(loss_ce2.size()[0]/32)

        loss_ce1 = loss_ce1.reshape((32,l1_s))
        loss_ce2 = loss_ce2.reshape((32,l2_s))
        l1 = lab.repeat_interleave(l1_s).reshape(32,-1)
        l2 = lab.repeat_interleave(l2_s).reshape(32,-1)
        loss_ce1 = l1*loss_ce1
        loss_ce2 = (l2^1)*loss_ce2


        # print(lab)
        # print(loss_ce1)
        # print(loss_ce1.size())
        # print(loss_ce2)
        # print(loss_ce2.size())
        
        loss_ce1 = loss_ce1.mean()
        loss_ce2 = loss_ce2.mean()

        # kk = list(np.linspace(0.0, 1.0, num=len(train_iterator)))

        
        # if i_==len(iterator)-80:
        #     neg_humor_vec,pos_humor_vec = [],[]
        #     print('constructing style vectors...')
        #     for i in range(1000):
        #         style,lab = get_vectors(i,SRC,TRG,model,'cuda')
        #         if lab==0:
        #             neg_humor_vec.append(style)
        #         else:
        #             pos_humor_vec.append(style)


        #     print(len(neg_humor_vec))
        #     print(len(pos_humor_vec))
        #     print(neg_humor_vec[0])
        #     print(neg_humor_vec[0].size())


        #     neg_humor_vec = torch.stack(neg_humor_vec).squeeze(dim=1).mean(dim=0)
        #     pos_humor_vec = torch.stack(pos_humor_vec).squeeze(dim=1).mean(dim=0)

        #     print(neg_humor_vec.size())
        #     print(pos_humor_vec.size())
        #     torch.save(pos_humor_vec,'hum_vec.pt')


        if epoch_nos>=0:
            rl_loss = reinforce(batch.s,batch.t,SRC,TRG,model,device,batch.l)
            loss = 0.5*(loss_ce1+loss_ce2+loss_mse+0.0*loss_st)+0.5*rl_loss
            if i_%50==0:
                print(loss_ce1,loss_ce2,loss_mse,loss_st,mi,rl_loss)
            

            
        else:
            loss = loss_ce1+loss_ce2+loss_mse+0.0*loss_st
            if i_%50==0:
                print(loss_ce1,loss_ce2,loss_mse,loss_st,mi)
        
        
        
        
        
        # loss.backward()

        # # print(model.)
        
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # optimizer.step()
        model.backward(loss)
        model.step()
        
        epoch_loss += (loss_ce1+loss_ce2).item()
        
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.s
            trg = batch.t
            lab = batch.l

            output_h, _, output_nh,_, style_h,style_nh, content, pred_style,_ = model(src, trg[:,:-1])
            
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
            
            output_dim_h = output_h.shape[-1]
            
            output_h = output_h.contiguous().view(-1, output_dim_h)

            output_dim_nh = output_nh.shape[-1]
            
            output_nh = output_nh.contiguous().view(-1, output_dim_nh)

            trg = trg[:,1:].contiguous().view(-1)
            
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]

            result = torch.einsum('abc, abc->ab', style_h, style_nh)
            bs = style_h.size()[0]
            ll = style_h.size()[1]
            loss_mse = nn.MSELoss()(result, torch.zeros((bs,ll)).to(device))
            loss_ce1 = criterion(output_h, trg).mean()
            loss_ce2 = criterion(output_nh, trg).mean()
            # loss_st = criterion_style(pred_style,lab)
            
            

            epoch_loss += (loss_ce1+loss_ce2).item()
        
    return epoch_loss / len(iterator)



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to('cuda')
    
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, evaluate=False):
    if evaluate:
        d =  OneHotCategorical(logits=logits.view(-1, latent_dim, categorical_dim))
        return d.sample().view(-1, latent_dim * categorical_dim)

    y = gumbel_softmax_sample(logits, temperature)
    return y.view(-1, latent_dim * categorical_dim)

def translate_sentence_for_baseline(sentence, src_field, trg_field, model, device,lab, max_len = 50):
    
    # model.train()
        
    if isinstance(sentence, str):
        # nlp = spacy.load('de_core_news_sm')
        nlp = en_core_web_sm.load()
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = tokens[0:98]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    remaining = 100-len(tokens)

    tokens = tokens+[src_field.pad_token]*remaining

    # print(tokens)
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    # print(src_indexes)

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    # print(src_tensor)
    
    src_mask = model.make_src_mask(src_tensor)
    
    # with torch.no_grad():
    enc_src = model.encoder(src_tensor, src_mask)
    style_h = model.style_proj_h(enc_src)
    style_nh = model.style_proj_nh(enc_src)

    pred_style = model.style_labeler(torch.mean(style_h,dim=1))

    content = model.content_proj(enc_src)

    #style and content's resultant should reconstruct input sentence,so,
    # enc_out = style+content

   
    enc_out_h = style_h+content
    enc_out_nh = style_nh+content

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    pred_toks = []
    W = []
    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        
        # with torch.no_grad():
        if lab==0:
            output, attention, sample, w_grad = model.decoder_humor(trg_tensor, enc_out_h, trg_mask, src_mask,gumbel_softmax=True)
        if lab==1:
            output, attention, sample, w_grad = model.decoder_nonhumor(trg_tensor, enc_out_nh, trg_mask, src_mask,gumbel_softmax=True)
        
        # gumbel_one_hot = F.gumbel_softmax(output[:,-1,:].squeeze(),tau=1,hard=True)
        # ll = int(list(gumbel_one_hot.size())[0])
        # aranged = torch.arange(ll).to('cuda')
        # hard_sample_w_grad = torch.sum(aranged*gumbel_one_hot) # reparameterization to get one hot vector
        # # print(hard_sample_w_grad)
        # # token_probs = F.softmax(output[:,-1,:],dim=1) # softmax prob distribution
        # # m = Categorical(token_probs)
        # # action = m.sample()
        # # print(action)
        w_grad = w_grad.unsqueeze(0)
        pred_token = output.argmax(2)[:,-1].item()
        # print(sample)
        # print(hard_sample_w_grad,pred_token)
        # pred_toks.append(hard_sample_w_grad.unsqueeze(0))
        pred_toks.append(sample)
        W.append(w_grad)
        # print(sample,w_grad)
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    pred_toks = torch.cat(pred_toks).unsqueeze(0)

    W = torch.cat(W).unsqueeze(0)
    # print(pred_toks)
    # pred_toks = pred_toks.type(torch.LongTensor)
    # print(pred_toks)
    return trg_tokens[1:], attention, pred_toks, W




def translate_sentence(sentence, src_field, trg_field, model, device,lab, max_len = 50):
    
    model.eval()
    # print('*********')
    # print(sentence)
    if isinstance(sentence, str):
        # nlp = spacy.load('de_core_news_sm')
        nlp = en_core_web_sm.load()
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = tokens[0:98]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    remaining = 100-len(tokens)

    tokens = tokens+[src_field.pad_token]*remaining

    # print(tokens)
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    # print(src_indexes)

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    # print(src_tensor)
    
    src_mask = model.make_src_mask(src_tensor)
    
    # with torch.no_grad():
    enc_src = model.encoder(src_tensor, src_mask)
    style_h = model.style_proj_h(enc_src)
    style_nh = model.style_proj_nh(enc_src)

    pred_style = model.style_labeler(torch.mean(style_h,dim=1))

    content = model.content_proj(enc_src)

    #style and content's resultant should reconstruct input sentence,so,
    # enc_out = style+content

   
    enc_out_h = style_h+content
    enc_out_nh = style_nh+content

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    pred_toks = []
    W = []
    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
          if lab==0:
            output, attention, sample, w_grad = model.decoder_humor(trg_tensor, enc_out_h, trg_mask, src_mask,gumbel_softmax=True)
          if lab==1:
            output, attention, sample, w_grad = model.decoder_nonhumor(trg_tensor, enc_out_nh, trg_mask, src_mask,gumbel_softmax=True)
        
        
        w_grad = w_grad.unsqueeze(0)
        pred_token = output.argmax(2)[:,-1].item()
        # print(sample)
        # print(hard_sample_w_grad,pred_token)
        # pred_toks.append(hard_sample_w_grad.unsqueeze(0))
        pred_toks.append(sample)
        W.append(w_grad)
        # print(sample,w_grad)
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    pred_toks = torch.cat(pred_toks).unsqueeze(0)

    W = torch.cat(W).unsqueeze(0)
    # print(pred_toks)
    # pred_toks = pred_toks.type(torch.LongTensor)
    # print(pred_toks)
    return trg_tokens[1:], attention, pred_toks, W









def eval(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            # src = batch.s
            # trg = batch.t
            # lab = batch.l
            list_of_src_tokens = batch.s
            list_of_trg_tokens = batch.t
            src_field = SRC
            trg_field = TRG
            model = model
            device = 'cuda'
            lab_ = batch.l
            max_len = 50
            n_training_episodes = int(batch.s.size()[0])
            print('n training episode {}'.format(n_training_episodes))
            # reinforce(batch.s,batch.t,SRC,TRG,model,device,batch.l)



            for i_episode in tqdm(range(n_training_episodes)):
              
              lab = lab_[i_episode].item()
              saved_log_probs = []
              rewards = []
              # sentence = list_of_src_sentences[i_episode]
              # print(sentence)
              # trg_sentence = list_of_trg_sentences[i_episode]
              # if isinstance(sentence, str):
              #     # nlp = spacy.load('de_core_news_sm')
              #     nlp = en_core_web_sm.load()
              #     tokens = [token.text.lower() for token in nlp(sentence)]
              # else:
              #     tokens = [token.lower() for token in sentence]

              s = []
              for i in list_of_trg_tokens[i_episode,:]:
                  
                  
                  if TRG.vocab.itos[i.item()]=='<pad>':
                      # print('reached')
                      break
                  s.append(TRG.vocab.itos[i.item()])

                  # print(len(s))
                  
              trg_sentence = ' '.join(s[1:-1])
              # print(trg_sentence)

              s = []
              for i in list_of_src_tokens[i_episode,:]:
                  
                  
                  if SRC.vocab.itos[i.item()]=='<eos>':
                      # print('reached')
                      break
                  s.append(SRC.vocab.itos[i.item()])

                  # print(len(s))
                  
              sentence = s[1:]
              sentence = ' '.join(sentence)
              print('***********')
              print(sentence)
              baseline,_,bs_toks,bs_toks_soft = translate_sentence(sentence, SRC, TRG, model, device,lab, max_len = 50)
              # print('bs_toks {}'.format(bs_toks))
              print(' '.join(baseline[:-1]), lab)
              print('*********')








def get_vectors(i, src_field, trg_field, model, device, max_len = 50):


    sentence = vars(train_data.examples[i])['s']
    lab = vars(train_data.examples[i])['l']
    model.eval()
            
    if isinstance(sentence, str):
        # nlp = spacy.load('de_core_news_sm')
        nlp = en_core_web_sm.load()
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = tokens[0:98]
    
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]


    remaining = 100-len(tokens)
    tokens = tokens+[src_field.pad_token]*remaining
    
    # print(tokens)
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    # print(src_indexes)

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    # print(src_tensor)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
        style = model.style_proj(enc_src)

        

    return style,lab

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    optimizer=None
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP,epoch)
    valid_loss = evaluate(model, test_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    # if valid_loss < best_valid_loss:
    #     best_valid_loss = valid_loss
    torch.save(model.state_dict(), './double_decoder_rl_cc_yelp_remaining/tr_model_full_reward_epoch{}.pt'.format(epoch))
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')



























# for example_idx in range(4,500):


#     src = vars(test_data.examples[example_idx])['s']
#     trg = vars(test_data.examples[example_idx])['t']
#     lab = vars(test_data.examples[example_idx])['l']

#     if lab!=1:
#         print(f'src = {src}')
#         print(f'trg = {trg}')

#         translation_, attention = translate_sentence(src, SRC, TRG, model, device, pos_humor_vec, neg_humor_vec)


#         print(f'predicted pos trg (1.5) = {translation_}')

#         translation_, attention = translate_sentence(src, SRC, TRG, model, device, pos_humor_vec, neg_humor_vec,grade=1.0)


#         print(f'predicted pos trg (1.0) = {translation_}')


#         translation_, attention = translate_sentence(src, SRC, TRG, model, device, pos_humor_vec, neg_humor_vec,grade=2.0)


#         print(f'predicted pos trg (2.0) = {translation_}')

#         translation, attention = translate_sentence(src, SRC, TRG, model, device, pos_humor_vec, neg_humor_vec, is_pos=False)

#         print(f'predicted trg = {translation}')

#         print('**********************')









