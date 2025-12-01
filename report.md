Architecture Engineering Report: Stabilizing Recursive Reasoning
This report summarizes the iterative engineering process used to transform the Tiny Recursive Model (TRM) from an unstable prototype into a robust reasoning engine capable of solving both Logic (Parity) and Memory (Associative Recall) tasks.
🎯 Executive Summary
The core challenge was training a recursive loop effectively 96 layers deep (12 steps × 8 sub-layers). Standard techniques failed due to the Stability-Plasticity Dilemma: the model either exploded (learning nothing) or stagnated (refusing to change state).
The Final Solution: A Gated Recurrent Architecture (GRU-style) combined with Backpropagation Through Time (BPTT) and Pre-Normalization.
1. The Foundation: Fixing the Pipeline
Initial State: The original implementation was fast but mathematically flawed for reasoning tasks.
The Fix: True BPTT (Backpropagation Through Time).
What we did: Instead of stepping the optimizer inside the loop (breaking gradients), we accumulated gradients across all 12 steps and stepped once at the end.
Why: This allowed the model to link Step 12’s failure back to Step 1’s decision.
Outcome: The model could theoretically learn, but immediately hit deep learning bottlenecks.
2. Solving "Amnesia" & "Vanishing Gradients"
The Problem: The model failed Parity (Loss 0.69) and Recall (Loss 3.0+). It was guessing randomly.
Attempt A: Positional Embeddings
Why: The model was permutation-invariant (Bag-of-Words). It couldn't distinguish "Start" from "End."
Outcome: Essential for logic, but revealed the next issue: signal death.
Attempt B: Residual Connections (x = x + y)
Why: Signals were dying before traveling through 96 layers (Vanishing Gradient).
Outcome: Explosion (Loss 13.0+). The signal traveled, but kept growing unchecked until weights hit infinity.
Attempt C: Gradient Clipping
Why: To cap the explosion.
Outcome: Stabilized training, but the internal state was still chaotic.
3. The Stability-Plasticity War (The Hardest Phase)
The Problem: We needed the model to be stable enough to hold memory (Recall) but flexible enough to flip bits (Parity).
Attempt D: Fixed Scaling (0.1 * update)
Why: To dampen the explosion caused by residuals.
Outcome: Stagnation. Recall passed (Stable), but Parity failed (Too rigid). The model couldn't "snap" its state fast enough for sharp logic.
Attempt E: Removing Scaling
Why: To allow fast updates.
Outcome: Immediate Explosion (Loss 30+). Without brakes, the deep recurrence amplified noise exponentially.
Attempt F: Learnable Scaling (LayerScale)
Why: Let the model decide when to brake.
Outcome: Bias towards Stability. The model learned to be a "Vault" (perfect Memory), but refused to open up for Logic updates. Parity failed.
4. The Final Architecture: Gated Recurrence
The Problem: We cannot tune a single scalar to satisfy both Memory (Keep State) and Logic (Change State).
The Solution: Gated Updates (The "GRU" Trick)
Math: 
h
n
e
w
=
(
1
−
z
)
⋅
h
o
l
d
+
z
⋅
update
h 
new
​
 =(1−z)⋅h 
old
​
 +z⋅update
Why it works: It introduces a learnable gate (
z
z
) for every neuron.
For Recall: The gate closes (
z
≈
0
z≈0
), preserving memory perfectly.
For Parity: The gate opens (
z
≈
1
z≈1
), allowing instant state flips.
Outcome: Success on both tasks. The model no longer explodes (bounded by 0-1) and no longer stagnates (gates can fully open).
✅ Final Winning Configuration
Component	Setting	Reason
Recurrence	Gated (Sigmoid)	Solves Stability vs. Plasticity conflict.
Normalization	Pre-Norm	Creates a clean gradient superhighway.
Embeddings	Positional	Required for order-dependent logic.
Optimizer	AdamW	Standard, with clip_grad_norm_(1.0).
Learning Rate	1e-3	High enough to learn fast, safe because of Gating.
Weight Decay	0.0	Critical. Non-zero decay erases memory in recursive loops.
