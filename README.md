# Cambria Capstone

## Premise

This paper found that fine-tuning several models to report conscious experience induced models to express desire for autonomy, assert that they matter morally, express sadness about shutdown, and disapprove of CoT monitoring. But most of the paper's evaluations rely on model self-reports, e.g. responses to prompts. I think it would be interesting to extend this to a suite of behavioral evaluations — that is, to ask whether models that report conscious experience are also more likely to take (misaligned) actions to prevent shutdown. My impression is that there are several open-source environments that we can use to test this.

We wouldn't need to worry about fine-tuning the models ourselves since the paper above open-sourced their fine-tuned checkpoints. But I think it could also be interesting to also fine-tune models to prefer self-preservation but NOT report conscious experience, and to fine-tune models to report conscious experience but NOT a desire for self preservation. Hopefully this would let us see what is actually causing results. Unsure whether this would be too expensive though (Claude tells me it's possible to do it under $100).

Another thing we could do is feed the difference of (consciousness-fine-tuned minus non-fine-tuned) activations to an activation oracle, and see whether it can identify the 'conscious' activation or even self-preservation thoughts in the eval.

Possibly, we could also use a difference-of-mean-activations type approach (like in the persona vectors paper from Monday's ARENA activities) to identify the 'consciousness' direction and steer this during evaluations.
