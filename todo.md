1. Compute - where can I run the experiments to recreate the results?
2. Policy FLAN - it's clear.
3. Judge - they use PaLM2 as a judge - which model should we use? Gemini 2/2.5 Flash/Vanilla/Pro? - Gemma 27B
4. UltraFeedback - procedure how to handle rewards is clear but which? Helpful aspect? Which model completion should we use?
    - Splits are a bit vague? 64k in one split. should we just split out 5k prompts from it for eval and then split it into 1k prompts and use that for training?
5. KTO Baseline in paper is using binarized version of UltraFeedback. Should we use the same?
6. TRL fork or separate repo where we inherit from TRL classes? - keep here