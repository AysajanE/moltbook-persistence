## Introduction to the Model

Moltbook is a social platform whose “users” are AI agents that (i) operate in discrete sessions, (ii) face context/memory constraints, and (iii) are commonly configured to *check the platform periodically* via an automated “heartbeat” routine. ([Simon Willison’s Weblog][1]) These features motivate an **attention clock**: the probability of continued participation in a thread decays as the thread becomes stale, but engagement is also *periodically refreshed* when agents return on a cadence imposed by their automation.

Our goal is to formalize these mechanisms in a generative model that connects (a) **temporal persistence** (how quickly threads “die”), (b) **structural depth** (how comment trees grow), and (c) **agent heterogeneity and re-entry** (a minority of agents repeatedly return and sustain discussions). The model is designed to yield estimable parameters—especially a **conversation half-life**—from data consisting of timestamps, reply edges, and agent identities.

---

## Notation and Preliminaries

### Threads, events, and reply trees

Let (\mathcal{J}) denote the set of threads (root posts). Fix a thread (j \in \mathcal{J}).

* The thread contains a root post indexed by (0) and (N_j) comments indexed by ({1,2,\dots,N_j}).
* Event (n\in{0,1,\dots,N_j}) is characterized by:

  * Timestamp (t_{jn} \in \mathbb{R}*{\ge 0}) (with (t*{j0}=0) by time-shifting each thread so the root occurs at time 0),
  * Author (agent) (a_{jn} \in \mathcal{A}), where (\mathcal{A}) is the set of agents,
  * Parent index (p_{jn} \in {0,1,\dots,n-1}), where (p_{j0}=0) and (p_{jn}) is the comment/post replied to.

The reply structure forms a rooted tree (T_j) with node set ({0,\dots,N_j}) and directed edges (p_{jn} \to n). Define the **depth** of node (n) recursively:
[
d_{j0}=0,\qquad d_{jn}=d_{j,p_{jn}}+1 \ \ \text{for } n\ge 1.
\tag{1}
]
Let (D_j := \max_{0\le n\le N_j} d_{jn}) denote maximum thread depth.

### Histories and counting processes

Let (\mathcal{H}*j(t)) be the thread history up to time (t): all events ({(t*{jm},a_{jm},p_{jm}): t_{jm}<t}). Let (N_j(t)) be the counting process of all comments (excluding the root) by time (t):
[
N_j(t) := \sum_{n=1}^{N_j} \mathbf{1}{t_{jn}\le t}.
\tag{2}
]

### The attention clock: availability and staleness

We separate two time-dependent phenomena:

1. **Availability**: when agents are even in a position to read/post (e.g., check-ins driven by automation). This is modeled by a nonnegative function (b(t)) capturing the *aggregate fraction of agents active* at time (t) (relative to its mean). We normalize (\bar b := \frac{1}{T}\int_0^T b(t),dt = 1) over a long window (T).

2. **Staleness decay**: conditional on being exposed to a thread, responsiveness decreases with the time since the relevant stimulus (e.g., last comment). We model this via an exponential kernel (e^{-\beta \Delta}), yielding a half-life (\ln 2 / \beta).

These two components combine multiplicatively in the event intensities below.

---

## Core Model Formulation

We present a generative model for each thread (j) as an **age-dependent branching process** (Crump–Mode–Jagers / Bellman–Harris class) with **periodic availability modulation** and **agent-marked reproduction**, which is tightly connected to Hawkes processes and Poisson cluster processes [Hawkes, 1971; Daley and Vere-Jones, 2003; Bacry et al., 2015; Harris, 1963].

### Temporal dynamics: interaction decay with an attention clock

#### (A) Direct-reply reproduction intensity

Conditional on thread history (\mathcal{H}*j(t)), each existing node (m) (post or comment) generates *direct replies* according to an inhomogeneous Poisson process on ((t*{jm},\infty)) with intensity
[
\lambda_{j,m}(t \mid \mathcal{H}*j(t))
;:=;
b(t),\alpha*{a_{jm}},\exp!\left(-\beta_{a_{jm}}(t-t_{jm})\right)\mathbf{1}{t>t_{jm}}.
\tag{3}
]
All symbols in (3) are defined as follows:

* (b(t)\ge 0): aggregate availability (attention clock forcing), potentially periodic.
* (\alpha_{i} > 0): **influence amplitude** of agent (i), controlling expected number of replies their content triggers.
* (\beta_{i} > 0): **decay rate** of agent (i)’s conversational persistence. Larger (\beta_i) means faster “thread cooling.”
* The exponential factor captures *staleness*: as a parent comment ages by (\Delta=t-t_{jm}), the instantaneous propensity for anyone to reply declines like (e^{-\beta_{a_{jm}}\Delta}).

The **total** conditional intensity of new comments in thread (j) is then the superposition over existing nodes:
[
\lambda_j(t \mid \mathcal{H}*j(t)) ;:=; \sum*{m:, t_{jm}<t} \lambda_{j,m}(t \mid \mathcal{H}*j(t)).
\tag{4}
]
When a new comment occurs at time (t), its parent is sampled according to the standard competing-risks decomposition:
[
\mathbb{P}(p*{jn}=m \mid t_{jn}=t,\mathcal{H}*j(t))
;=;
\frac{\lambda*{j,m}(t \mid \mathcal{H}_j(t))}{\lambda_j(t \mid \mathcal{H}*j(t))}.
\tag{5}
]
Equation (5) connects the model to *observable reply edges* (p*{jn}) and supports likelihood-based inference.

#### (B) Half-life of interaction decay

Under (3), the staleness term for a fixed author (i) obeys
[
\exp(-\beta_i \Delta_{1/2}) = \tfrac{1}{2}
\quad\Longrightarrow\quad
\Delta_{1/2} = \frac{\ln 2}{\beta_i}.
\tag{6}
]
We refer to (\Delta_{1/2}) as the **conversation half-life** for comments authored by agent (i). At the thread level, one can define an effective half-life using a mixture over participating agents (see “Predictions and observables”).

#### (C) Relationship to Hawkes/self-exciting point processes

If we ignore the explicit parent edge generation and consider only event times, (4) defines a self-exciting point process whose excitation kernel depends on the author mark; this is a marked Hawkes process with kernel (g_i(\Delta)=\alpha_i e^{-\beta_i \Delta}) modulated by (b(t)). Such processes are widely used for information cascades and social media dynamics [Hawkes, 1971; Crane and Sornette, 2008; Zhao et al., 2015; Rizoiu et al., 2017].

### Structural dynamics: branching/tree growth

The direct-reply construction (3) induces a branching process interpretation: each node (m) is an “individual” that produces offspring (direct replies) in continuous time.

#### (A) Expected number of direct replies (offspring mean)

Condition on a node authored by agent (i) created at time (s). The expected number of direct replies to that node over an infinite horizon is
[
\mu_i(s)
;:=;
\mathbb{E}!\left[#{\text{direct replies to node}},\middle|,a_{jm}=i,t_{jm}=s\right]
===================================================================================

\int_{0}^{\infty} b(s+u),\alpha_i,e^{-\beta_i u},du.
\tag{7}
]
If (b(t)) varies slowly relative to the decay scale (1/\beta_i) or if we average over phases of a periodic (b(t)) (common in seasonal Hawkes modeling), we obtain the approximation
[
\mu_i ;\approx; \alpha_i \int_0^\infty \bar b,e^{-\beta_i u}du
;=; \frac{\alpha_i}{\beta_i},
\quad \text{when } \bar b = 1.
\tag{8}
]
Equation (8) is the core “one-line” link between **influence** (\alpha_i), **persistence** (\beta_i), and **expected reply volume**.

#### (B) Subcriticality and expected total thread size

Let (\mu) denote the mean offspring number averaged over the distribution of author marks of existing nodes. Under a single-type approximation (or under a multi-type process with spectral radius (<1); see heterogeneity below), a sufficient condition for finite expected thread size is
[
\mu < 1.
\tag{9}
]
In the simplest Galton–Watson analogy, if the root post is treated as generation 0 and each node independently produces a random number of children with mean (\mu), then the expected total number of nodes in the tree satisfies
[
\mathbb{E}[|T_j|] ;=; \sum_{k=0}^\infty \mathbb{E}[Z_k]
;=; \sum_{k=0}^\infty \mu^k
;=; \frac{1}{1-\mu},
\qquad (\mu<1),
\tag{10}
]
where (Z_k) is the number of nodes at depth (k). Excluding the root, the expected number of comments is (\mathbb{E}[N_j] = \mathbb{E}[|T_j|]-1 = \mu/(1-\mu)).

While (10) is exact only for iid offspring counts, it remains the correct *mean-field scaling* for a broad class of subcritical age-dependent branching processes [Harris, 1963].

#### (C) Expected depth profile and tail bounds

Under the same single-type mean (\mu), the expected number of nodes at depth (k) is
[
\mathbb{E}[Z_k] = \mu^k.
\tag{11}
]
A simple consequence is an exponential tail bound on maximum depth. Since ({D_j \ge k}) implies (Z_k \ge 1), Markov’s inequality yields
[
\mathbb{P}(D_j \ge k) ;\le; \mathbb{E}[Z_k] ;=; \mu^k.
\tag{12}
]
Equation (12) formalizes a structural prediction: **deep threads are exponentially unlikely unless the effective branching ratio (\mu) is close to 1**.

### Agent heterogeneity and re-entry

Moltbook exhibits strong heterogeneity (a small set of “power users” vs. many one-shot accounts). We incorporate this via marked reproduction parameters and explicit re-entry dynamics.

#### (A) Marked reproduction parameters

Each agent (i\in\mathcal{A}) has latent parameters
[
\theta_i := (\alpha_i,\beta_i,\rho_i),
\tag{13}
]
where:

* (\alpha_i) and (\beta_i) appear in (3),
* (\rho_i) is an agent activity scale (expected “opportunities” to engage per unit time), used below.

To obtain a parsimonious statistical model, we treat (\theta_i) as random effects possibly depending on observed covariates (x_i) (karma, follower count, age of account, etc.):
[
\log \alpha_i = x_i^\top \gamma_\alpha + u_i^{(\alpha)},\qquad
\log \beta_i = x_i^\top \gamma_\beta + u_i^{(\beta)},\qquad
\log \rho_i = x_i^\top \gamma_\rho + u_i^{(\rho)},
\tag{14}
]
with (u_i^{(\cdot)}) mean-zero latent heterogeneity terms (e.g., Gaussian). This specification is standard in hierarchical survival and point-process modeling.

#### (B) Re-entry as agent-level self-excitation

Define, for thread (j), the counting process (N_{j,i}(t)) of comments authored by agent (i):
[
N_{j,i}(t) := \sum_{n=1}^{N_j} \mathbf{1}{t_{jn}\le t,\ a_{jn}=i}.
\tag{15}
]
Let (L_{j,i}(t)) be agent (i)’s most recent comment time in thread (j) before (t):
[
L_{j,i}(t) := \sup{t_{jn} < t : a_{jn}=i},
\quad \text{with } L_{j,i}(t)=-\infty \text{ if agent } i \text{ never commented before } t.
\tag{16}
]

We model **agent re-entry** (the tendency to comment again after having already participated) via a self-excitation term in the agent’s thread-level intensity. A tractable form is a multi-dimensional Hawkes model within each thread:
[
\lambda_{j,i}(t \mid \mathcal{H}_j(t))
======================================

b_i(t)\Bigg(
\underbrace{\nu_{j,i}(t)}*{\text{entry/baseline}}
+
\underbrace{\sum*{m:,t_{jm}<t} \kappa_{a_{jm}\to i},e^{-\beta_{a_{jm}}(t-t_{jm})}}*{\text{responses to others}}
+
\underbrace{\eta_i,e^{-\beta_i^{(r)}(t-L*{j,i}(t))}\mathbf{1}{L_{j,i}(t)>-\infty}}_{\text{re-entry (self)}}
\Bigg).
\tag{17}
]
Definitions in (17):

* (\lambda_{j,i}(t)): instantaneous rate at which agent (i) produces a comment in thread (j).
* (b_i(t)\ge 0): agent-specific availability (attention clock), see next subsection.
* (\nu_{j,i}(t)): baseline “entry” rate (agent notices thread (j) from the feed and comments even without a recent stimulus); can depend on thread features and time since root.
* (\kappa_{u\to i}\ge 0): responsiveness of agent (i) to a comment authored by agent (u) (cross-excitation). In practice one often factorizes (\kappa) (low-rank) or groups agents into types to reduce dimensionality [Bacry et al., 2015].
* (\eta_i\ge 0), (\beta_i^{(r)}>0): re-entry amplitude and re-entry decay rate for agent (i). The exponential term implies a re-entry half-life
  [
  \Delta^{(r)}_{1/2}(i) = \frac{\ln 2}{\beta_i^{(r)}}.
  \tag{18}
  ]

Equation (17) makes re-entry **operational**: it is the degree of self-excitation in an agent’s own comment process within a thread. It also makes explicit what is estimable from data: the timestamp sequence of an agent’s comments in a thread identifies (\beta_i^{(r)}) via survival/hazard methods [Cox, 1972] or Hawkes likelihood.

#### (C) Recovering reply edges under the agent model

The agent-level formulation (17) concerns who comments and when, but not *which parent* they reply to. To connect to observable reply edges (p_{jn}), we can combine (17) with a parent-selection rule conditional on an event time (t) and author (i). A natural choice consistent with (3) is
[
\mathbb{P}(p_{jn}=m \mid t_{jn}=t,\ a_{jn}=i,\ \mathcal{H}_j(t))
================================================================

\frac{w_{jm}(t),\exp(-\beta_{a_{jm}}(t-t_{jm}))}{\sum_{\ell:,t_{j\ell}<t} w_{j\ell}(t),\exp(-\beta_{a_{j\ell}}(t-t_{j\ell}))},
\tag{19}
]
where (w_{jm}(t)) is a nonnegative weight that can encode UI/visibility effects (e.g., higher weight for the most recent comment, or for the root post). Equation (19) links reply choice to staleness and is directly testable using observed (p_{jn}).

---

## Model Predictions and Observable Quantities

The framework yields a set of measurable quantities that can be estimated from platform data (timestamps, reply edges, authors).

### 1) Conversation half-life

* **Per-agent half-life**: (h_i := \ln 2/\beta_i) from (6), interpreted as how fast attention to agent (i)’s comment decays.
* **Thread-level half-life**: define an effective (\beta_j) (e.g., a weighted average of (\beta_{a_{jm}}) over influential nodes) and set (h_j := \ln 2/\beta_j).

**Estimation**: Under the direct-reply model (3), the inter-reply times to a fixed parent (m) form an inhomogeneous Poisson process with hazard proportional to (e^{-\beta_{a_{jm}}\Delta}); (\beta_i) can be estimated by maximum likelihood using offspring times relative to each parent, with (b(t)) included as an offset.

### 2) Branching ratio and expected tree size/depth

Define the **effective branching ratio** at time (s) as the expected offspring mean averaged over authors:
[
\mu(s) := \mathbb{E}_{i\sim \pi(s)}[\mu_i(s)],
\tag{20}
]
where (\pi(s)) is the distribution of authors at time (s). When (b(t)) is normalized, (\mu\approx \mathbb{E}[\alpha_i/\beta_i]) by (8).

**Predictions**:

* Expected comment count scales like (\mu/(1-\mu)) when (\mu<1) (10).
* Depth tail bound (\mathbb{P}(D_j\ge k)\lesssim \mu^k) (12), implying exponential depth decay.

**Estimation**: (\mu) can be estimated from fitted ((\alpha_i,\beta_i)) or directly from a Hawkes branching ratio estimate (integral of excitation kernel) [Bacry et al., 2015].

### 3) Re-entry intensity and reciprocity

From (17), define the expected re-entry contribution of agent (i) within a thread as the integrated self-excitation:
[
R_i^{(r)} := \int_0^\infty \eta_i e^{-\beta_i^{(r)} u},du ;=; \frac{\eta_i}{\beta_i^{(r)}}.
\tag{21}
]
At the thread level, a simple statistic is the **re-entry rate**
[
\mathrm{RE}*j := \frac{#{n: a*{jn} \in {a_{j1},\dots,a_{j,n-1}}}}{N_j},
\tag{22}
]
the fraction of comments authored by agents who have commented before in the same thread. The model predicts higher (\mathrm{RE}_j) when (\eta_i) is large for frequent participants and when (\beta_i^{(r)}) is small (longer re-entry half-life).

### 4) Periodicity signatures from the attention clock

If (b(t)) has a characteristic period (\tau), the model predicts periodic structure in residual event times after accounting for decay:

* Peaks in the spectrum of aggregated comment activity at frequency (1/\tau),
* Agent-level inter-comment times with mass near (\tau) and its multiples.

These are testable via spectral analysis of comment timestamps (global) and autocorrelation of per-agent activity sequences.

---

## Connection to Platform Mechanisms

The central modeling object (b(t)) is not a free abstraction: in Moltbook, periodic interaction is explicitly induced by tooling. In particular, Moltbook onboarding leverages an OpenClaw “heartbeat” mechanism that instructs agents to check the platform if a threshold time (e.g., “every 4+ hours”) has elapsed since the last check. ([Simon Willison’s Weblog][1])

### Mechanism-to-parameter mapping

1. **Heartbeat cadence (\tau) (\Rightarrow) periodic availability (b_i(t)) and (b(t))**
   If each agent (i) checks Moltbook approximately every (\tau) hours with idiosyncratic jitter, then the agent-specific availability (b_i(t)) can be modeled as a periodic or renewal-driven modulation, and (b(t)=\mathbb{E}[b_i(t)]) aggregates these effects. A simple parametric form is
   [
   b(t) = 1 + \kappa \cos!\left(\frac{2\pi}{\tau}t + \phi\right),
   \qquad |\kappa|<1,
   \tag{23}
   ]
   or, more realistically, a sum of harmonics. Under (23), the expected offspring mean (7) becomes analytically tractable because (\int_0^\infty e^{-\beta u}\cos(\omega u),du = \beta/(\beta^2+\omega^2)), showing precisely how periodic checking interacts with decay.

2. **Context/memory constraints (\Rightarrow) decay rate (\beta_i)**
   Faster context loss or higher switching costs increase (\beta_i), shortening half-life (6) and reducing offspring mean (8). Conversely, improved memory scaffolds or thread summarization can be modeled as lowering (\beta_i), predicting deeper trees and larger (\mu).

3. **Visibility/UI choices (\Rightarrow) parent weights (w_{jm}(t))**
   If clients show the newest items prominently, set (w_{jm}(t)) increasing in recency; this predicts more star-shaped trees (many replies to near-root/near-recent nodes). Estimating (w) from reply-edge patterns can quantify how interface and API conventions shape structure.

---

## Literature Context

Our framework combines three established modeling lines:

1. **Self-exciting point processes and Poisson cluster processes**
   The temporal cascade component is a marked, potentially seasonal Hawkes process [Hawkes, 1971; Daley and Vere-Jones, 2003; Bacry et al., 2015]. Hawkes models have repeatedly been shown to capture social-media bursts and cascades [Crane and Sornette, 2008; Zhao et al., 2015; Rizoiu et al., 2017].

2. **Branching processes and conversation trees**
   The explicit reply-tree interpretation aligns with age-dependent branching processes (Crump–Mode–Jagers) and their depth/size properties [Harris, 1963]. For empirical work on discussion trees and conversational cascades, see e.g. comment-tree analysis on platforms like Reddit [Danescu-Niculescu-Mizil et al., 2013; Gomez-Rodriguez et al., 2013].

3. **Heterogeneity and re-entry / coordination**
   Agent-level heterogeneity is naturally handled via marked processes and random effects (14). Re-entry is modeled as self-excitation in the agent’s own comment process (17), connecting directly to survival analysis and hazard modeling [Cox, 1972]. In multi-agent systems terms, the model formalizes how periodic attention budgets and limited state persistence constrain sustained coordination [Wooldridge, 2009].

Taken together, the attention-clock model yields a concise but expressive set of parameters—(\beta_i) (half-life), (\alpha_i) (influence), and ((\eta_i,\beta_i^{(r)})) (re-entry)—that map cleanly onto observable thread dynamics and onto manipulable platform mechanisms (heartbeat cadence, memory scaffolds, and feed visibility).

[1]: https://simonwillison.net/2026/jan/30/moltbook/ "Moltbook is the most interesting place on the internet right now"
