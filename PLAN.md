FT-PFN uses a random acquisition function: MFPI-Random. The idea is to replace this with a different approach. The approach is basically:
for a pair of given partial trainings with different hyperparams, we can calculate the probability of wether one is going to beat the other over a time period t:
p(training a beats training b) = sigmoid(score a - score b). in the datasets we use, we have the data for that. 
additionally, we think about the HPO as a sort of RL-like MDP: the state is the set of started training runs + n suggestions (of hyperparameters). the action is a one hot: which config/run to spend the next unit of compute/the next step on. 
a "step"/"unit of compute" in the MDP is either a fixed set of gradient steps in the training, or a fixed fraction of an epoch, or a fixed wall time. wall time is the most honest one. but that does not matter during training, only during deployment, since we assume our training data to represent all of these modalities well enough.
that means that we have O(number of started training runs + number of suggestions)^2 losses for each training step. 
we can then use these scores to define a policy: softmax(scores) = p(this is the best choice) under weak assumptions. 
the encoding works like this:
one token per run/suggestion + one time token: 
* a flag for a special general information token: which time horizon are we considering for the scores? how many steps (in the MDP) have passed, how many steps remain?
* a flag for wether it is a hyperparam suggestion or a run that has already been started
* in the already started runs, k dimensions encode the training curve so far with some characteristics of it. needs to be normalized _across all visible runs_ to make it loss scale-agnostic. we only care about relative performance anyways. these dimensions are zeroed out for suggestions. 
* in the already started runs, a dimension documents how many steps (in the MDP) this particular training has been run already. 
* m dimensions that just directly encode the hyperparameters. might need to be normalized as well. categorical hyperparams need to be handled with a one-hot encoding. 
e.g. 256 dims for token dimension is more than enough for a sensible amount of information about the training curve so far and leave enough dims for hyperparameters. 512 can be used as well if necessary, or any other number. but goal should be to be small-ish. 

an open question is how to encode time. In deployment, computational and time resources might differ drastically. so we might need to train with a wide variety of budgets, or we normalize the time dims. i lean towards normalization + discrete info about e.g. 10 last steps.

five more points: datasets, how to make hyperparam suggestions, DAgger-like training, "final policy & the horizon question", and an implementation detail

Datasets:

We can use the IFBO prior from FT-PFN, but just as well TaskSet and other datasets on NN training. i lean towards the IFBO prior for training and the other "real" datasets for eval, just like the FT-PFN paper did

How to make hyperparam suggestions: 

FT-PFN kinda cheated in my opinion: they passed _all_ points that are available in the TaskSet and other datasets to the transformer. So they do not provide in their paper an actual way to sample suggestions, IIRC. i might be wrong. 
random suggestions might work reasonably well, but the better suggestions we make, the less steps our model spends on worse hyperparameters/runs.
I think i didnt make one thing clear enough yet: in my idea, _every MDP step gets n new, randomly sampled hyperparameter suggestions_. that's like a rougelike's shop. that means a NEW SEED for each step. that is basically a very stochastic environment. the question i want to discuss is: what probability density do we sample those from, and how big is n? maybe very large is the right answer even. and maybe random uniform or normal or mutation + uniform is a good enough place to start. or maybe even use the network itself as a generator for a gradient on langevin dynamics xdd. but that might maybe be better as a second version experiment. Or maybe some cheap surrogate like a TPE. 

DAgger-like training: 
I said p(this is the best choice) = softmax(pairwise-trained scores). The question is how to use that as a policy: my intuition is to _not_ be greedy, and rather use the softmax scores directly as a multinomial to sample from. the reason i mentioned DAgger is ONLY that my intuition is to use this _same_ policy to actually steer the MDP during training. this whole thing is basically RL with pseudo-labels (i will tell you later why pseudo) for the policy. and we should do it in parallel across multiple "environments". that is the ONLY reason in mentioned DAgger, DO NOT import any other tricks or things or specifics from that paper. my first intuition is to just use one "environment" per batch dim, no "replay buffer"-like data structure or anything. 

final policy and the horizon question: 
this ties into the "pseudo" labels. The problem is that if we put the full remaining training time into the global info token, the actual probability we measure is "how probable is it, that this will be the best run, if we spend _all_ the remaining time on it?". the problem is that we _won't_ be spending all remaining time on it with very high probability. because of the stochastic policy, because new info will change the picture, or because better hyperparameter suggestions might pop up later. i still say the right thing to do is use the softmax here as a multinomial, not greedily, but even if we used it greedily, the decision might flip in a later MDP state under more information and/or other new hyperparam suggestions. i think of 2 knobs here: temperature scaling the softmax and tweaking/underreporting the remaining time horizon that we put into the global info token. for the second one, the idea is: "we wont _really_ end up spending all of the remaining time on that token, so we might as well underrepresent that time". and if we think of training phases, later on, with little time remaining, it will be not sensible to start a new run, and the network will know that. that thought experiment kind of suggests that closer to the end, we should put the _whole_ remaining time into the global token. idk, that is up for experimentation.

a specific about the training harness for real world deployment:
no matter what kind of unit of compute we end up using, the whole thing is first and foremost designed to train one network at a time, with the freedom to freeze and thaw runs at will. to that end, for any real world deployment, we need a harness that automatically caches a training state away from the GPU, onto ram or disk. that means _ONLY THE ACTIVE RUN SHOULD LIVE ON THE GPU IN THE REAL WORLD DEPLOYMENT CASE_

Summary: this is almost like a rougelike deckbuilding game: the cards are the started runs, and the shop is the suggestions. the shop is different on each step. you decide: what do i spend my next budget on? do i continue to train one of my cards in hand, or do i start to train one from the shop, adding it to my deck? naming comes very natural: rouge + dagger fits well together. Rouge&Dagger HPO? something like that. 

I provided you with the FT-PFN paper and its repo (both in private/ifBO), which contains the ifBO prior. any additional material like taskset etc. you can search online and download yourself. this is a shared development server with other researchers, leave at least 2 GPUs free at any time, and use at most 3 GPUs at the same time. 
your task is to implement this idea into code. to that end, the first steps are to repo within the working folder rouge-dagger, and adding a readme and a backlog.
training should be logged on wandb.
after implementing and training, compare its performance to FT-PFN.

there exists a prior version of this work which used RL instead of the O(n^2)+dagger training on this server in the automl-exam folder. it can be used as rough reference but shouldn't be relied upon too much.

venv is at /shared/work/venv
