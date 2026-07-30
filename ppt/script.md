# Speaker Script — Pre-Demo Presentation

**NYC Taxi Fare & Trip Duration Prediction** · Anyone AI · ML Developer Career

Open `ppt/index.html` in a browser and press **F** for fullscreen.
Use `→` / `Space` to advance. You can jump straight to a slide with `index.html#14`.

---

## Timing at a glance

| # | Speaker | Section | Slides | Words | Time |
|---|---------|---------|--------|-------|------|
| 1 | **Keyneth Lara** | Introduction · Motivation · Team · Overview · Dataset | 1 – 11 | 353 | 2:49 |
| 2 | **William Vera** | Phase 1 — Foundation | 12 – 13 | 138 | 1:06 |
| 3 | **Keyneth Lara** | Phase 2 — Data | 14 – 16 | 161 | 1:17 |
| 4 | **Marcos Villabasa** | Phase 3 — Modelling | 17 – 19 | 164 | 1:18 |
| 5 | **Mauricio Mora** | Phase 4 — Serving | 20 – 22 | 177 | 1:24 |
| 6 | **Néstor Mamani** | Phase 5 — Delivery + Phase X | 23 – 26 | 165 | 1:19 |
| | | | **26 slides** | **1,158** | **9:15** |

Times are measured, not estimated, at **125 words per minute** — a calm speaking
pace. That leaves **45 seconds** inside the 10-minute limit.

> **Read this before you rehearse.** Forty-five seconds is not much. Groups
> almost always run slower live than on paper. So: rehearse once with a timer,
> and if you land over 9:30, use the cut list at the end of this file. Do not try
> to save time by speaking faster — speaking faster is what makes a presentation
> hard to follow.

> Keyneth speaks twice (4:06 in total) because he carries the whole introduction
> and Phase 2. Everyone else is close to the 1:30 target. That is the agreed
> running order.

---

## How to read this script

- Written at **B1 level**: short sentences, common words.
- `[SLIDE n]` tells you when to press the arrow key.
- **Bold** = say this a little louder, or slow down.
- `(…)` = a stage direction, not something you say.
- Big numbers are written the way you should **say** them.

---

# 1 · Keyneth Lara

**Slides 1 – 11 · about 2 minutes 25 seconds**

### [SLIDE 1] Title

Good morning, everyone. We are the NYC Taxi team.
My name is Keyneth.
Today we want to ask you two very simple questions.
**How much?** And **how long?**

### [SLIDE 2] 115,747

This is New York.
Every single day, more than **one hundred fifteen thousand** people get into a yellow taxi.
But here is the strange part.
Almost none of them know the price before the trip starts.

### [SLIDE 3] A city that runs on the meter

Three and a half million rides — in only **one month**.
Around thirteen thousand taxis. One point six billion dollars a year.
And look at this curve.
At four in the morning, the city sleeps.
At six in the evening, it explodes. **Thirteen times** more rides.

### [SLIDE 4] The rules just changed

For years, the meter decided the price.
You learned the total only at the end.
Then Uber and Lyft arrived. They show you the price **first**.
Now New York lets yellow taxis do the same.
But to give a price **before** the ride, you have to predict it.

*(pause — this is the reason the project exists)*

### [SLIDE 5] Predict at the door

So this is our problem.
When the passenger closes the door, we know only a few things.
The time. The pickup zone. The drop-off zone. The passengers.
From that, we predict the **fare** in dollars, and the **duration** in minutes.

### [SLIDE 6] One prediction, three winners

Why does that matter?
The passenger gets trust.
The driver sees which ride really pays.
The company gets better prices and better planning.

### [SLIDE 7] The team

We are six people from four countries.
Each of us presents one part today.

### [SLIDE 8] Nineteen tickets, five phases

Our plan is nineteen tickets in five phases.
Foundation. Data. Modelling. Serving. Delivery.

### [SLIDE 9] One month. Three files.

Now, the data. One month — May 2022. Three files.
The trips: three and a half million rows.
A table that gives every zone a name.
And a shapefile with the real map.

### [SLIDE 10] There are no coordinates

Here is our first surprise. There are **no coordinates**.
New York removed latitude and longitude in 2016.
We only get a zone number, from one to two hundred sixty-five.
So we build the coordinates ourselves — from this map.

### [SLIDE 11] Half the columns are cheating

Our second surprise is bigger.
Half of the columns are **cheating**.
The tip, the total, the drop-off time — all of that exists only
**after** the ride ends. So we cannot use it.
Only seven columns are honest.

→ *Hand off:* "William will show you how we started."

---

# 2 · William Vera

**Slides 12 – 13 · about 1 minute 10 seconds**

### [SLIDE 12] Phase 01 — Foundation

Thank you, Keyneth.
I am William, and I will talk about phase one: the foundation.

### [SLIDE 13] Build the rails before the train

By foundation, I mean all the work we did before we start any training work.

First, the repository structure.
We have two separate Python trees. One trains the model. The other serves it. They never import each other.

Second, research.
We read papers about what other people did, then we check which ideas still work with the 2022 data.

Third, the download of the information, which will be done with our data ingestion script.
For this script, we want it to be reproducible, so no matter where we run, we always end up with the same input data in all cases.

→ *Hand off:* "Keyneth will take you into the data."

---

# 3 · Keyneth Lara

**Slides 14 – 16 · about 1 minute 15 seconds**

### [SLIDE 14] Phase 02 — Data

Thanks, William.
Phase two is the data. This is where the project is won or lost.

### [SLIDE 15] What 3.6 million rides told us

We explored three point six million rides. Three things surprised us.

First, look at the picture on the left.
Almost every ride costs about ten dollars. But a few cost thousands.
That shape is very bad for a model.
On the right, after a simple log transform, it looks normal again.

Second — cleaning is not a small step. Cleaning **is** the model.
In the raw file, distance and fare have almost no correlation. Zero point zero one.
After we remove the impossible rows, it jumps to **zero point nine five**.

Third, one record says a taxi drove at seven million miles per hour.

### [SLIDE 16] From raw rows to one fitted object

So we clean, we split, and we build features.
And we split by **time**, never at random.
We train on the first three weeks and we test on the last week.
A random split would let the model see the future.

→ *Hand off:* "Marcos, over to you."

---

# 4 · Marcos Villabasa

**Slides 17 – 19 · about 1 minute 25 seconds**

### [SLIDE 17] Phase 03 — Modelling

Thank you, Keyneth.
I am Marcos, and this is phase three: the modelling.

### [SLIDE 18] Three families, run in parallel

We do not pick one model and hope.
We run three families at the same time.

The first is baselines. We start with something very simple —
always predict the average. Every other model has to beat that.

The second is gradient boosting: LightGBM and XGBoost.

The third is a neural network — a multi-layer perceptron.

All three start from the same features.
All three use the same splits and the same metrics.
So the comparison at the end is honest.

### [SLIDE 19] Accuracy is only half the score

Then we choose. And here is the important part.
We do **not** choose only the most accurate model.
We also measure speed.
A model that is a little better but much slower is not useful,
because it has to answer a live request.

And the final file has to survive alone.
It carries the model, the preprocessing, and the order of the features.
It must run without our training code.

→ *Hand off:* "Mauricio will show you why."

---

# 5 · Mauricio Mora

**Slides 20 – 22 · about 1 minute 25 seconds**

### [SLIDE 20] Phase 04 — Serving

Thanks, Marcos.
I am Mauricio, and this is phase four: serving.

### [SLIDE 21] One artifact is the whole interface

This is our complete system.

On the left is the host. We prepare the data, and we train.
That produces one file: **model dot pkl**.

That file is shared with the container.
It is the **only** connection between the two sides.

Inside the container we run a FastAPI service.
It loads the model once, when it starts — not on every request.

It gives three endpoints.
Predict — you send zone IDs and a time, and you get the fare and the minutes.
Health — it tells you if the model is up.
And docs, for the contract.

Pydantic checks the input.
If a zone number is wrong, you get a clear message, never a crash.

### [SLIDE 22] Make it real, then make it hold

Then we make it solid.
A dashboard, where you pick two zones and a time, and see the answer on a map.
Tests — including one that fails if a forbidden column reaches the model.
A latency budget, with real measurements.
And Docker, so it runs anywhere with one command.

*(that is the whole system — pause before handing over)*

→ *Hand off:* "Néstor will close."

---

# 6 · Néstor Mamani

**Slides 23 – 26 · about 1 minute 20 seconds**

### [SLIDE 23] Phase 05 — Delivery

Thank you, Mauricio.
I am Néstor, and I will close with phase five: delivery.

### [SLIDE 24] Show it to someone who can break it

Two things here.

First, peer review.
We give the project to another team, on a clean machine, and we watch them run it.
We write down every problem they find.
Then we decide: fix now, later, or never.
The "fix now" list must be empty before we finish.

Second, documentation.
Every step becomes an exact command, so anyone can repeat our results.
And we write our limits honestly.
One month of data. And distance is an approximation.

### [SLIDE 25] Only if it earns its place

We also have two optional ideas.
Weather — does rain change the ride?
And demand — where will the next ride be?
But we only add them if they really help.
They must never put the main project at risk.

### [SLIDE 26] Closing

So — **how much**, and **how long**?

Today we showed you the problem, the data, and our plan.
Next time, we will answer both questions **live**, with a real prediction.

Thank you very much.

*(pause, then open for questions)*

---

## Delivery notes

**Pace.** The script is written for about 125 words per minute. That is a calm,
clear speed. If you feel rushed, you are going too fast — the slides have very
few words, so the audience is listening to you, not reading.

**Hand-offs.** Say the next person's name, and look at them. They start with
"Thank you, *name*". That single line makes the six of us feel like one team.
Practise only the hand-offs once — they are where most groups lose time.

**Numbers.** Say them the simple way. "One hundred fifteen thousand" is enough —
you do not need "seven hundred forty-seven". The exact number is on the slide.

**Pauses.** There are three places to stop for one second:
after "you have to predict it" (slide 4), after "cleaning **is** the model"
(slide 15), and after the last question on slide 26.

**If you are running late.** Cut these, in this order:
1. Slide 7 — the team slide. Just say "we are six people from four countries."
2. Slide 25 — the optional ideas. Say one sentence: "we have two optional ideas,
   only if they help."
3. Slide 6 — say only "the passenger gets trust, the driver gets a better choice."

**If a laptop fails.** The deck is one HTML file with everything inside it. Any
browser, no internet needed except for the fonts. Copy `ppt/` to a USB stick and
it still works.
