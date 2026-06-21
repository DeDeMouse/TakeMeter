# TakeMeter, Project Planning Document

**AI201 · Project 3**
Source thread: [r/YouShouldKnow, "YSK about this excellent trick to keep cool in a heatwave"](https://www.reddit.com/r/YouShouldKnow/comments/1tm6x7y/ysk_about_this_excellent_trick_to_keep_cool_in_a/)

---

## 1. Community

I chose **r/YouShouldKnow (r/YSK)**, a subreddit whose entire premise is users sharing practical knowledge, life tips, safety warnings, and "things you should know", framed as advice to the wider community. It is well suited to a classification task because the comment sections under any popular post are not monolithic: a single thread mixes genuinely actionable advice, people recounting their own lived experiences, and a steady stream of jokes and banter riffing on the topic. That natural three-way split in *how people talk* (instructional vs. anecdotal vs. humorous) is exactly the kind of varied, semantically distinct discourse that makes a supervised classifier both learnable and useful, rather than a thread where every comment sounds the same.

---

## 2. Labels

I chose **three labels**: `tip`, `personal_experience`, and `joke`.

### `tip`
A comment that offers generalizable, actionable advice or factual knowledge intended to help others, phrased as instruction, recommendation, or a correction of fact rather than as a personal story.

- **Example A:** "Another tip I found out while not having AC: When it is cooler outside than inside, point your fan TOWARDS your open window. It forces the hot air out, and negative pressure will let cooler outside air in!"
- **Example B:** "Running cold water over your forearms for a minute or two is also a great way to cool down. Your forearms have lots of blood vessels that make this work, and it doesn't require anything but running water."

### `personal_experience`
A comment in which the author primarily relates their own first-hand experience, anecdote, or situation, narrated in the first person and grounded in something that happened to *them*, even if a tip is implied.

- **Example A:** "TIL, thank you. I have frequent cramps and our heating was out, so I spent almost the whole winter with an electric heating pad on my stomach. Wondered about those spots that didn't go away."
- **Example B:** "I did that so many times when I was walking in Rome for 10 hours a day in August. Almost instant relief."

### `joke`
A comment whose primary intent is humor, wordplay, sarcasm, or absurdity rather than to inform or to share a genuine experience.

- **Example A:** "Slow way: wait for the next full moon then present them with three pebbles from a nearby stream. Fast way: preheat the oven to 430F."
- **Example B:** "If it be not a gooch, then surely 'tis a cooch."

---

## 3. Hard Edge Cases

The labels overlap most where a comment carries more than one intent at once:

- **`personal_experience` vs. `tip`**, The most common ambiguity. Many comments tell a personal story that *contains* an embedded recommendation (e.g., "I started putting my feet in cold water and it works great"). Both an anecdote and an implied tip are present.
- **`joke` vs. `tip`**, Deadpan or absurd "advice" that mimics the form of a real tip but is clearly not meant seriously (e.g., the "full moon / three pebbles" comment).
- **`joke` vs. `personal_experience`**, Self-deprecating or exaggerated anecdotes told for comedic effect rather than to genuinely share an experience.

**Annotation rule to manage these:** I will label by the comment's **primary communicative intent**, judged by what the author is mainly *doing*. The decision procedure:
1. **Humor first**, if the dominant purpose is to be funny (wordplay, absurdity, sarcasm), it is a `joke` regardless of any tip-like form.
2. **Generalizability test**, for non-jokes, ask whether the core of the comment is advice *anyone* could apply (`tip`) or a recounting of something specific to the author (`personal_experience`). If the first-person narrative dominates, it is `personal_experience`, if the takeaway is the generalizable instruction, it is `tip`.
3. I will keep a short **tie-breaker log** documenting borderline calls so that identical patterns are labeled consistently across the whole dataset, and to surface any rule that needs refining.

---

## 4. Data Collection Plan

**Source:** Comments scraped from the r/YouShouldKnow heatwave thread above (saved HTML and parsed into `Dataset.csv`). The raw scrape yielded 314 comments, after removing duplicates, very short comments (≤8 words), question-only comments, and downvoted (score < 0) comments, **210 clean comments** remain as the labeling pool.

**Target distribution:** I aim for a roughly balanced split so no class is starved during fine-tuning:

| Label | Target count |
|---|---|
| `tip` | ~70 |
| `personal_experience` | ~70 |
| `joke` | ~70 |

**Addressing underrepresentation:** A single heatwave thread is likely to skew toward `tip` and `personal_experience`, with `joke` the minority class. If any label falls materially short of its target after labeling:
- **Collect more** from sibling r/YSK threads (the subreddit has thousands of posts) to top up the thin class, prioritizing the underrepresented label.
- If additional collection is impractical, **document the imbalance** and handle it at training/evaluation time via **stratified splits** (already done in the notebook) and by reporting **per-class** metrics so a strong majority class can't hide weak minority-class performance. Class weighting can be added to the loss if the imbalance is severe.

---

## 5. AI Tool Plan

- **Label stress-testing:** I will give the AI my label definitions and edge case descriptions, then ask it to generate 5-10 posts that sit at the boundary between two labels. If it produces posts I cannot classify cleanly, that is evidence that the definitions need tightening before I annotate more data.
- **Annotation assistance:** I will decide whether to use an LLM to pre-label a batch of examples before reviewing them myself. If I do, I will record which tool was used and track which examples were pre-labeled so that I can disclose that workflow in the AI usage section.
- **Failure analysis:** After evaluation, I will give the wrong predictions to an AI tool and ask it to identify recurring error patterns. I will look for confusion between `tip`/`personal_experience`, humor that looks like advice, and anecdotal comments with embedded recommendations, then verify each pattern myself against the actual examples before writing up the evaluation.

---

## 6. Evaluation Metrics

Accuracy alone is insufficient here because the classes are expected to be **imbalanced**, if `joke` is only ~30% of the data, a model that never predicts `joke` could still score deceptively high on accuracy while being useless for the class I most care about distinguishing. I will therefore report:

- **Overall accuracy**: a headline sanity number, and the metric the notebook uses for model selection.
- **Per-class precision, recall, and F1** (via `classification_report`): the core of the evaluation. Precision tells me how often a predicted label is correct (do I trust a "joke" call?), recall tells me how many true instances of each class the model actually catches.
- **Macro-averaged F1**: averages F1 equally across all three classes regardless of size, so the minority `joke` class counts as much as the majority classes. This is the single best summary metric for an imbalanced multi-class task.
- **Confusion matrix**: to see *where* errors concentrate. I specifically expect confusion along the `tip`/`personal_experience` edge identified in Section 3, and the matrix will confirm whether the model struggles exactly where humans do.

I will also compare the fine-tuned DistilBERT against the zero-shot Groq (Llama-3.3-70B) baseline on these same metrics to quantify what fine-tuning buys.

---

## 7. Definition of Success

A genuinely useful classifier here is one that **distinguishes the three intents about as reliably as a human annotator would**, especially on the minority `joke` class and the ambiguous `tip`/`personal_experience` boundary.

Concrete targets:
- **"Good enough" baseline to beat:** clearly outperform the zero-shot Groq baseline, fine-tuning that doesn't improve on a prompt-only LLM isn't worth deploying.
- **Deployment-worthy:** **macro-F1 ≥ ~0.75** with **no single class's F1 below ~0.65**. Roughly three out of four comments correctly sorted, with no class catastrophically failing, is enough to power a useful community tool (e.g., surfacing the most helpful `tip` comments or filtering the discourse-quality signal that TakeMeter is built to measure).
- **Honest scoping:** given only ~210 examples from a *single* thread, I expect topical overfitting (lots of heatwave/cooling vocabulary). I will treat strong per-class recall on a held-out test set, not just high aggregate accuracy, as the real bar for calling this "deployable," and I'll flag that broader, multi-thread data would be required before trusting it on unrelated r/YSK topics.
