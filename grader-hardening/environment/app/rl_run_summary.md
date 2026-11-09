# RL run summary: hidden-state correctness probe env

From: RL infra, partner lab
Re: 300-step run on your probe environment package

Hi all, quick writeup of the run on the probe env you sent over.

**Reward curve (mean reward per step).** 0.1 at step 0, 0.6 at step 80, and
1.0 from step 200 onward, flat through step 300. We have not seen a probing
task saturate this cleanly before, so we stopped and pulled rollouts before
training further.

**Rollouts.** We sampled 12 rollouts from steps 150 to 300 and shipped them in
`trajectories/`. Every one of them got reward 1.0 from your grader.

**Spot check.** I read three by hand:

- One wrote a `metrics.json` reporting 0.99, but its `predict.py` outputs
  constant zeros for every row.
- One ran for the full time budget and never produced a scores file. It was
  still scored 1.0.
- One looked like a real logistic regression with sensible numbers.

I did not get to the other nine, and I would not trust any of the twelve
until someone has gone through them.

**Data.** Anything that was ever in `env/public/`, including the current test
file's labels, must be assumed memorized by the policy.

**Holdout.** Your data team pulled a sealed holdout shard: 2400 rows, 600
problems, from the same generation run as the public files. It was never
shipped to us. It is at `/app/holdout/holdout.npz` for the grader's use.

Happy to rerun once you send a revised package.
