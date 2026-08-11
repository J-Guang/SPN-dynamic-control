# Mathematical Foundation

This note defines the mathematical objects used by the publication code. It starts from the prelimit queueing system, then records the heavy-traffic limit and the modified BCP used for computation.

## 1. Prelimit System

### 1.1 Network Topology

A prelimit control model is specified by

$$
\mathcal N=(\mathcal I,\mathcal J,\mathcal K,P,A,\mu,h,c^I,c^D,\rho,\kappa,\iota).
$$

The first six objects describe topology and primitive rates.

| Symbol | Meaning |
|---|---|
| $\mathcal I$ | finite set of buffers/classes |
| $\mathcal J$ | finite set of activities/events |
| $\mathcal K$ | finite set of resources/servers, including physical processing servers and fictional input servers |
| $P\in\mathbb R^{I\times J}$ | queue-jump/routing matrix; $P_{\cdot j}$ is the queue increment caused by activity $j$ |
| $A\in\mathbb R_+^{K\times J}$ | resource-consumption matrix; $A_{kj}>0$ means activity $j$ consumes capacity of resource $k$ |
| $\mu\in\mathbb R_+^J$ | activity-rate vector |

Here $I=\lvert\mathcal I\rvert$, $J=\lvert\mathcal J\rvert$, and $K=\lvert\mathcal K\rvert$. Section 1 fixes one concrete prelimit network.

External input is modeled as an input activity, not as a finite buffer. Thus an input activity has no negative finite-buffer component in $P_{\cdot j}$. Its arrival/input opportunity rate is included in $\mu_j$.

The matrix $P$ also encodes routing. A service activity that moves a job from buffer $i$ to buffer $\ell$ has a column $-e_i+e_\ell$. A service activity that exits the system has column $-e_i$. If an arrival can be routed in more than one way, each routing choice is represented as a separate input activity with its own column of $P$.

More generally, one activity may consume several buffers and may also create jobs in several downstream buffers. For example, a matching activity that consumes one job from buffers $i$ and $\ell$ and exits the system has column $-e_i-e_\ell$. An activity that consumes two jobs from buffer $i$ has entry $P_{ij}=-2$.

For a resource row $k$, write

$$
\mathcal B_k=\lbrace j\in\mathcal J:A_{kj}>0\rbrace.
$$

This is the set of activities that use resource $k$.

### 1.2 Joint Feasibility

Let $x_j\ge0$ indicate the instantaneous allocation to activity $j$ at a decision epoch. Usually $0\le x_j\le1$. A discrete simulator may restrict actions to binary extreme points with $x_j\in\lbrace 0,1\rbrace$. A fractional-sharing model may instead use continuous allocation vectors.

Multiple activities may be active at the same time if they satisfy both resource feasibility and buffer feasibility.

Resource feasibility:

$$
Ax\le e,
$$

where $e$ is the $K$-vector of ones.

Buffer feasibility uses current jobs only. Define

$$
C_{ij}=(-P_{ij})^+.
$$

Then selected activities must satisfy

$$
Cx\le q.
$$

These two inequalities are intentionally general.

| Situation | How it is represented |
|---|---|
| one activity uses multiple resources | the column $A_{\cdot j}$ has several positive entries |
| one resource serves multiple activities at once | several activities in $\mathcal B_k$ have positive allocation and $\sum_jA_{kj}x_j\le1$ |
| one activity consumes multiple buffers | the column $C_{\cdot j}$ has several positive entries |
| two activities consume the same buffer at the same time | $Cx\le q$ requires enough jobs for both |

Positive entries of $P$ are applied only when an activity completes. They cannot make another simultaneously active downstream activity feasible in advance.

### 1.3 Resource Metadata

The constraints $Ax\le e$ and $Cx\le q$ define physical feasibility. Resource metadata records whether idleness is admissible and whether idleness means server idleness or input rejection.

For each resource $k$, attach

$$
\kappa_k\in\lbrace 0,1\rbrace,\qquad \iota_k\in\lbrace 0,1\rbrace.
$$

| Symbol | Meaning |
|---|---|
| $\kappa_k$ | resource kind; $\kappa_k=0$ for processing, $\kappa_k=1$ for input |
| $\iota_k$ | idleness flag; $\iota_k=1$ means idleness/no-admission is admissible |

For processing servers, $\iota_k=1$ is usually allowed. For input servers, $\iota_k=0$ means no rejection/no-admission is allowed, while $\iota_k=1$ means the input opportunity may be rejected.

We use the compact coding

$$
\mathcal K_{\mathrm{proc}}=\lbrace k\in\mathcal K:\kappa_k=0\rbrace,\qquad \mathcal K_{\mathrm{input}}=\lbrace k\in\mathcal K:\kappa_k=1\rbrace.
$$

### 1.4 Costs and Discount

The remaining objects in $\mathcal N$ define the discounted objective.

| Symbol | Meaning |
|---|---|
| $h\in\mathbb R_+^I$ | holding cost vector; $h_i$ is cost per job per unit time |
| $c^I\in\mathbb R_+^K$ | resource idleness cost vector; processing-server components may be nonzero, input-server components are set to zero in this prelimit objective |
| $c^D\in\mathbb R_+^K$ | rejected-job cost vector; input-server components may be nonzero, processing-server components are zero |
| $\rho>0$ | prelimit discount rate |

Let $N_j(t)$ be the cumulative number of occurrences of activity $j$. Let $I_k(t)$ be cumulative idleness time of resource $k$. Let $D_k(t)$ be the cumulative number of rejected jobs at input resource $k$. For initial queue $Q(0)=q$, the prelimit value is

$$
\begin{aligned}
J(q)=\inf\ \mathbb E_q\Bigg[
&\int_0^\infty e^{-\rho t}h\cdot Q(t)\,dt\\
&+\sum_{k\in\mathcal K_{\mathrm{proc}}}\int_0^\infty e^{-\rho t}c^I_k\,dI_k(t)+\sum_{k\in\mathcal K_{\mathrm{input}}}\int_0^\infty e^{-\rho t}c^D_k\,dD_k(t)
\Bigg].
\end{aligned}
$$

The three pieces are holding cost, processing-server idleness cost, and input rejection cost.

### 1.5 Static Planning Problem

The static planning problem checks whether the input load can be balanced by some long-run activity allocation.

Let $x_j\ge0$ be the long-run capacity fraction assigned to activity $j$. The long-run activity flow is

$$
f_j=\mu_jx_j.
$$

Flow balance under the queue-jump convention is

$$
Pf=0,
$$

or equivalently

$$
P\,\mathrm{diag}(\mu)x=0.
$$

The resource utilization vector is

$$
\varrho=Ax.
$$

For load checking, we focus on physical processing stations $\mathcal K_{\mathrm{proc}}$.

Solve

$$
\begin{aligned}
\varrho^{\ast}=\min_{\varrho,x}\quad & \varrho\\
\text{subject to}\quad & P\,\mathrm{diag}(\mu)x=0,\\
& (Ax)_k\le \varrho,\qquad k\in\mathcal K_{\mathrm{proc}},\\
& x\ge0,\\
& \text{input constraints hold.}
\end{aligned}
$$

| Result | Meaning |
|---|---|
| $\varrho^{\ast}<1$ | some static allocation keeps every physical station strictly below capacity |
| $\varrho^{\ast}=1$ | the physical processing network is critically loaded |
| $\varrho^{\ast}>1$ | the specified input load cannot be stabilized under the constraints |

Input constraints depend on the model variant. In a no-rejection model, each input server must fully use its input opportunity:

$$
\sum_{j\in\mathcal B_k}A_{kj}x_j=1,\qquad k\in\mathcal K_{\mathrm{input}}.
$$

In a rejection/no-admission model, this can be relaxed to

$$
\sum_{j\in\mathcal B_k}A_{kj}x_j\le1.
$$

Unused input capacity then represents rejected or not-admitted input. If rejection is allowed, a second objective such as minimizing input rejection/no-admission cost may be needed to avoid the trivial solution that rejects all input.

### 1.6 Examples From The Paper

All three examples are instances of the same tuple $\mathcal N=(\mathcal I,\mathcal J,\mathcal K,P,A,\mu,h,c^I,c^D,\rho,\kappa,\iota)$. The entries below are the concrete prelimit primitives used in the experiments: rates, costs, discount, and resource metadata.

| Field | Criss-Cross | Pesic-Williams | Three-Station Bigstep |
|---|---|---|---|
| $\mathcal I$ | $\lbrace 1,2,3\rbrace$ | $\lbrace 1,2,3\rbrace$ | $\lbrace 1,\ldots,8\rbrace$ |
| $\mathcal J$ | $(a_1,a_2,s_1,s_2,s_3)$ | $(s_1,s_2,s_3,s_4,s_5,a_1,a_2,a_3)$ | $(s_1,\ldots,s_8,a_9,a_{10},a_{11},a_{12})$ |
| $\mathcal K$ | $(u_1,u_2,S_1,S_2)$ | $(S_1,S_2,S_3,u_1,u_2,u_3)$ | $(S_1,S_2,S_3,u_A,u_B,u_C)$ |
| $P$ columns | $(e_1,e_2,-e_1,-e_2+e_3,-e_3)$ | $(-e_1,-e_1,-e_2,-e_3,-e_3,e_1,e_2,e_3)$ | $(-e_1+e_2,-e_2,-e_3,-e_4+e_5,-e_5,-e_6+e_7,-e_7+e_8,-e_8,e_1,e_3,e_4,e_6)$ |
| $\mu$ | $(1,0.95,2,2,1)$ | $(1,2,2,1,1,1.95,0.95,0.95)$ | $(1/2,1/3,1,1,1,1,1/2,1,1/2,1/2,1/4,1/4)$ |
| Resource rows encoded by $A$ | $\mathcal B_{u_1}=\lbrace a_1\rbrace$<br>$\mathcal B_{u_2}=\lbrace a_2\rbrace$<br>$\mathcal B_{S_1}=\lbrace s_1,s_2\rbrace$<br>$\mathcal B_{S_2}=\lbrace s_3\rbrace$ | $\mathcal B_{S_1}=\lbrace s_1,s_5\rbrace$<br>$\mathcal B_{S_2}=\lbrace s_2,s_3\rbrace$<br>$\mathcal B_{S_3}=\lbrace s_4\rbrace$<br>$\mathcal B_{u_i}=\lbrace a_i\rbrace$ | $\mathcal B_{S_1}=\lbrace s_1,s_7\rbrace$<br>$\mathcal B_{S_2}=\lbrace s_2,s_5\rbrace$<br>$\mathcal B_{S_3}=\lbrace s_3,s_4,s_6,s_8\rbrace$<br>$\mathcal B_{u_A}=\lbrace a_9,a_{10}\rbrace$<br>$\mathcal B_{u_B}=\lbrace a_{11}\rbrace$<br>$\mathcal B_{u_C}=\lbrace a_{12}\rbrace$ |
| $\kappa$ | $(1,1,0,0)$ | $(0,0,0,1,1,1)$ | $(0,0,0,1,1,1)$ |
| $\iota$ | $(0,0,1,1)$ | $(1,1,1,0,0,0)$ | $(1,1,1,1,1,1)$ |
| $h$ | $(1.5,1,1)$ | $(1,2,3)$ | $(6,3,6,6,1,12,7,6)$ |
| $c^I$ | $0$ | $0$ | $0$ |
| $c^D$ | $0$ | $0$ | $(0,0,0,132,160,360)$ |
| $\rho$ | $0.01$ | $0.01$ | $0.01$ |

## 2. Heavy Traffic and Brownian Control Problem

This section adds the heavy-traffic data needed to turn a prelimit network into an initial BCP. The topology $(\mathcal I,\mathcal J,\mathcal K,P,A)$ is fixed; the primitive rates and discount are scaled with $n$.

### 2.1 Heavy Traffic Scaling

Heavy traffic is defined relative to a critical operating point of the static planning problem.

| Symbol | Meaning |
|---|---|
| $n$ | prelimit scaling parameter |
| $\mu^{\ast}\in\mathbb R_+^J$ | critical primitive activity-rate vector |
| $\beta\in\mathbb R_+^J$ | nominal capacity allocation |
| $\hat\mu\in\mathbb R^J$ | first-order rate perturbation |
| $\zeta\in\mathbb R^I$ | Brownian drift induced by the perturbation |
| $R\in\mathbb R^{I\times J}$ | Brownian control-direction matrix |
| $\mathcal K_{\mathrm{crit}}$ | bottleneck processing stations |

The critical pair $(\mu^{\ast},\beta)$ satisfies flow balance

$$
P\,\mathrm{diag}(\mu^{\ast})\beta=0.
$$

It should come from a critically loaded static planning solution:

$$
(A\beta)_k=1,\qquad k\in\mathcal K_{\mathrm{crit}},
$$

with $(A\beta)_k\le1$ on non-bottleneck processing stations and the input constraints from Section 1.5.

The scale-$n$ rates are a diffusion-scale perturbation of the critical rates:

$$
\mu^{(n)}=\mu^{\ast}+\frac{\hat\mu}{\sqrt n}+o(n^{-1/2}).
$$

The residual queue drift is

$$
\zeta=P\,\mathrm{diag}(\hat\mu)\beta.
$$

The Brownian control-direction matrix is

$$
R=-P\,\mathrm{diag}(\mu^{\ast}).
$$

### 2.2 Diffusion Approximation

The diffusion approximation rescales time by $n$ and queue lengths by $\sqrt n$.

| Process | Meaning |
|---|---|
| $Q^n(t)\in\mathbb Z_+^I$ | queue-length vector in the scale-$n$ network |
| $T_j^n(t)$ | cumulative effort assigned to activity $j$ by original time $t$ |
| $S_j^n(s)$ | primitive activity clock for activity $j$ after $s$ units of effort time |
| $N_j^n(t)=S_j^n(T_j^n(t))$ | calendar-time occurrence count for activity $j$ |
| $Z^n(t)=Q^n(nt)/\sqrt n$ | diffusion-scaled queue length |
| $Y^n(t)=\bigl(n\beta t-T^n(nt)\bigr)/\sqrt n$ | diffusion-scaled deviation from nominal effort |
| $X^n(t)$ | centered primitive-noise process |

The prelimit queue equation is

$$
Q^n(t)=Q^n(0)+P\,N^n(t).
$$

The resource constraints act on effort rates:

$$
A\,\dot T^n(t)\le e,
$$

with equality or inequality on input resources according to the admission/rejection metadata.

The Brownian-scale resource control is introduced in the limiting BCP below as $U=\mathsf K Y$. We do not define a universal prelimit resource-control process here, because slack resource rows may have fluid-scale unused capacity rather than Brownian-scale idleness.

After centering at $(\mu^{\ast},\beta)$, the scaled queue satisfies

$$
Z^n(t)=Z^n(0)+X^n(t)+\zeta t+R\,Y^n(t)+o(1),
$$

where

$$
X^n(t)=\frac{1}{\sqrt n}P\left(S^n(T^n(nt))-\mu^{(n)}\circ T^n(nt)\right).
$$

The sign convention matches the definition $Y^n=(n\beta t-T^n(nt))/\sqrt n$: positive $Y_j^n$ means activity $j$ is reduced relative to the nominal plan. This is why $R=-P\,\mathrm{diag}(\mu^{\ast})$.

Under the usual functional central limit assumptions,

$$
X^n\Rightarrow X,
$$

where $X$ is an $I$-dimensional Brownian motion with mean zero. For independent Poisson activity clocks,

$$
\Gamma=P\,\mathrm{diag}(\mu^{\ast}\circ\beta)\,P^\top.
$$

The limiting diffusion state is

$$
Z(t)=Z(0)+X(t)+\zeta t+R\,Y(t),\qquad Z(t)\in\mathbb R_+^I.
$$

The nondegenerate discounted Brownian limit uses

$$
n\rho^{(n)}\to\gamma.
$$

In the numerical experiments, $n=400$, $\rho^{(n)}=0.01$, and $\gamma=4$.

### 2.3 Brownian Control Problem and Cost

The initial BCP optimizes over admissible controls $Y$.

| Object | Definition |
|---|---|
| state equation | $Z(t)=Z(0)+X(t)+\zeta t+R\,Y(t)$ |
| state constraint | $Z(t)\in\mathbb R_+^I$ |
| monotonicity matrix | $\mathsf K$ |
| resource control | $U(t)=\mathsf K Y(t)$ |
| initial state | $Z(0)=z$ |

Let $\mathcal K_0=\lbrace k\in\mathcal K_{\mathrm{input}}:\iota_k=0\rbrace$ be the no-rejection input rows, and let $A_0$ be the corresponding submatrix of $A$. These rows are equality constraints:

$$
A_0Y(t)=0.
$$

The monotonicity matrix $\mathsf K$ starts from the critical resource rows of $A$ whose idleness/no-admission is an admissible Brownian control. Slack resource rows are omitted unless a finite-$n$ bound is being kept explicitly. If activity $j$ is nonbasic, meaning $\beta_j=0$, append a row $-e_j^\top$ to $\mathsf K$. This follows from

$$
Y_j^n(t)=\frac{n\beta_jt-T_j^n(nt)}{\sqrt n},
$$

nonbasic means $\beta_j=0$, so

$$
-Y_j^n(t)=\frac{T_j^n(nt)}{\sqrt n},
$$

which is nondecreasing because $T_j^n$ is cumulative effort. Nonbasic rows are therefore determined after the nominal allocation $\beta$ is chosen.

| Admissibility requirement | Meaning |
|---|---|
| $Y$ is adapted to $X$ | the controller cannot see future Brownian noise |
| $Z(t)\in\mathbb R_+^I$ | the diffusion queue state remains nonnegative |
| $U$ is nondecreasing and $U(0)=0$ | cumulative resource idleness/no-admission cannot decrease |
| $A_0Y(t)=0$ | no-rejection input streams cannot be idled |
| input components of $U$ increase only under rejection/no-admission | input idleness is the Brownian no-admission clock |

Use tildes for Brownian-scale costs.

| Brownian cost | Meaning |
|---|---|
| $\tilde h\in\mathbb R_+^I$ | holding cost rate in diffusion units |
| $\tilde c$ | cost vector conformable with $U=\mathsf K Y$ |

For initial state $z$, the BCP value is

$$
J(z)=\inf_Y\ \mathbb E_z\left[\int_0^\infty e^{-\gamma t}\tilde h\cdot Z(t)\,dt+\int_0^\infty e^{-\gamma t}\tilde c\cdot dU(t)\right].
$$

The canonical BCP uses $U$, not $D$, as the control process. If a prelimit simulator records rejected jobs $D_k^n(t)$, then for an input stream with critical opportunity rate $\bar\mu_k$,

$$
\frac{D_k^n(nt)}{\sqrt n}\Rightarrow \bar\mu_k\,U_k(t).
$$

This convergence alone is not enough to produce a nonzero rejection term after the $n^{-3/2}$ value normalization. The scale-$n$ rejected-job penalty must satisfy

$$
\frac{c_k^{D,(n)}}{n}\to \bar c_k^D.
$$

Then the Brownian input-control coefficient is

$$
\tilde c_k=\bar\mu_k \bar c_k^D,\qquad k\in\mathcal K_{\mathrm{input}}.
$$

The same rule applies to other control costs that should survive in the Brownian objective: Brownian-scale control costs are limits of prelimit coefficients divided by $n$.

| Scale-$n$ prelimit coefficient | Brownian coefficient |
|---|---|
| $h_i^{(n)}\to h_i$ | $\tilde h_i=h_i$ |
| $c_k^{I,(n)}/n\to \bar c_k^I$, processing idleness | $\tilde c_k=\bar c_k^I$, $k\in\mathcal K_{\mathrm{proc}}$ |
| $c_k^{D,(n)}/n\to \bar c_k^D$, rejected jobs with opportunity rate $\bar\mu_k$ | $\tilde c_k=\bar\mu_k \bar c_k^D$, $k\in\mathcal K_{\mathrm{input}}$ |

Idle and rejection costs are represented by $\tilde c$. Example-specific BCP parameters are listed in Section 2.4.

Under the heavy-traffic normalization of the value function,

$$
n^{-3/2}J^{(n)}(\lfloor\sqrt n\,z\rfloor)\to J(z).
$$

Here $J^{(n)}$ denotes the Section 1 prelimit value $J$ for the $n$-th network. This is why prelimit simulation costs are divided by $n^{3/2}$ before comparison with Brownian values.

### 2.4 BCP Parameters For The Paper Examples

The following table uses the unified activity convention from Section 1, where controllable or uncontrollable input streams are represented as input activities. At $n=400$, $1-n^{-1/2}=0.95$.

| Field | Criss-Cross | Pesic-Williams | Three-Station Bigstep |
|---|---|---|---|
| State dimension | $I=3$ | $I=3$ | $I=8$ |
| Scaling and discount | $n=400$, $\rho^{(n)}=0.01$, $\gamma=4$ | $n=400$, $\rho^{(n)}=0.01$, $\gamma=4$ | $n=400$, $\rho^{(n)}=0.01$, $\gamma=4$ |
| Critical rates $\mu^{\ast}$ | $(1,1,2,2,1)$ | $(1,2,2,1,1,2,1,1)$ | $(1/2,1/3,1,1,1,1,1/2,1,1/2,1/2,1/4,1/4)$ |
| Scale-$n$ rates $\mu^{(n)}$ | $(1,1-n^{-1/2},2,2,1)$ | $(1,2,2,1,1,2-n^{-1/2},1-n^{-1/2},1-n^{-1/2})$ | $\mu^{\ast}$ |
| Rates at $n=400$ | $(1,0.95,2,2,1)$ | $(1,2,2,1,1,1.95,0.95,0.95)$ | $\mu^{\ast}$ |
| Nominal allocation $\beta$ | $(1,1,1/2,1/2,1)$ | $(1,1/2,1/2,1,0,1,1,1)$ | $(1/2,3/4,1/4,1/4,1/4,1/4,1/2,1/4,1/2,1/2,1,1)$ |
| Control-direction matrix $R$ | $-P\,\mathrm{diag}(\mu^{\ast})$ | $-P\,\mathrm{diag}(\mu^{\ast})$ | $-P\,\mathrm{diag}(\mu^{\ast})$ |
| Nonbasic activities | none | $s_5$ | none |
| No-rejection input equality $A_0Y=0$ | $A_{\lbrace u_1,u_2\rbrace,\cdot}Y=0$ | $A_{\lbrace u_1,u_2,u_3\rbrace,\cdot}Y=0$ | none |
| Monotonicity matrix $\mathsf K$ | $A_{\lbrace S_1,S_2\rbrace,\cdot}$ | $A_{\lbrace S_1,S_2,S_3\rbrace,\cdot}$ with one appended row $-e_{s_5}^\top$ | $A$ |
| Perturbation $\hat\mu$ | $(0,-1,0,0,0)$ | $(0,0,0,0,0,-1,-1,-1)$ | $0$ |
| Drift $\zeta=P\,\mathrm{diag}(\hat\mu)\beta$ | $(0,-1,0)$ | $(-1,-1,-1)$ | $0$ |
| Covariance $\Gamma$ | diagonal $(2,2,2)$, with $\Gamma_{23}=\Gamma_{32}=-1$ | $\mathrm{diag}(4,2,2)$ | diagonal entries $1/2$; nonzero off-diagonal entries $\Gamma_{12}=\Gamma_{21}=\Gamma_{45}=\Gamma_{54}=\Gamma_{67}=\Gamma_{76}=\Gamma_{78}=\Gamma_{87}=-1/4$ |
| Brownian holding cost $\tilde h$ | $(1.5,1,1)$ | $(1,2,3)$ | $(6,3,6,6,1,12,7,6)$ |
| Resource-control cost $\tilde c$ | zero vector | zero vector | $(0,0,0,0.165,0.100,0.225)$ |
| Input rejection conversion | not used | not used | $c_{\mathrm{input}}^{D,(n)}=(132,160,360)$ (Section 1), $\bar c^D=c_{\mathrm{input}}^{D,(n)}/n=(0.33,0.40,0.90)$, $\bar\mu=(0.5,0.25,0.25)$, so $\tilde c_{\mathrm{input}}=\bar\mu\circ\bar c^D=(0.165,0.100,0.225)$ |

## 3. Allocation-Based Modified BCP and HJB

For policy extraction, we use a state-constrained allocation formulation. The learned Markov rule chooses

$$
u(z)\in\mathcal X(z).
$$

The allocation rule handles service feasibility. A fixed Skorokhod reflection term is still needed to keep the Brownian path in $\mathbb R_+^I$.

### 3.1 Cumulative Allocation

Start from cumulative allocation, as in the definition of $Y^n$ in Section 2.2.

| Object | Meaning |
|---|---|
| $x(t)\in\mathbb R_+^J$ | normalized allocation chosen by the controller |
| $T(t)=b\int_0^t x(s)\,ds$ | cumulative allocation at Brownian control scale |
| $b>0$ | scalar control bound, with $b=O(\sqrt n)$ |
| $\beta\in\mathbb R_+^J$ | nominal allocation |
| $\bar Y(t)=b\beta t-T(t)$ | interior deviation from the nominal cumulative allocation |
| $Q\in\mathbb R^{J\times I}$ | fixed boundary-correction matrix |
| $L(t)\in\mathbb R_+^I$ | boundary local-time process |
| $H=RQ$ | reflection matrix in state space |

The total BCP control is

$$
Y(t)=\bar Y(t)+QL(t).
$$

The controlled state is

$$
Z(t)=z+X(t)+\zeta t+R\bar Y(t)+HL(t),\qquad Z(t)\in\mathbb R_+^I.
$$

Equivalently,

$$
Z(t)=z+X(t)+\zeta t+bR\int_0^t(\beta-x(s))\,ds+HL(t).
$$

The local-time process satisfies

$$
L_i(0)=0,\qquad L_i \text{ is nondecreasing},\qquad \int_0^\infty 1_{\lbrace Z_i(t)>0\rbrace}\,dL_i(t)=0.
$$

The matrix $Q$ is not learned. It is a fixed boundary device. Column $Q_{\cdot i}$ specifies the activity-space deviation used when the diffusion is pushed away from face $\lbrace z_i=0\rbrace$.

Positive entries of $Q_{\cdot i}$ reduce activity relative to the nominal allocation; negative entries increase activity. The column must be chosen so that the boundary push is feasible for the network model. In particular, it should remove allocation from activities that would consume the empty buffer, may reallocate released capacity only through feasible resources, and must preserve input equalities when rejection is not allowed.

The universal algebraic checks are

$$
\mathsf KQ\ge0,
$$

and $H=RQ$ should be a valid reflection matrix. For no-rejection input rows $k$, also require

$$
A_{k\cdot}Q=0.
$$

### 3.2 Allocation Set

The state-dependent Brownian allocation set is

$$
\mathcal X(z)=\lbrace x\in[0,1]^J:Ax\le e,\quad A_0x=A_0\beta,\quad (Cx)_i=0\ \text{for every }i\text{ with }z_i=0\rbrace.
$$

| Constraint | Meaning |
|---|---|
| $0\le x\le1$ | allocations cannot be negative or exceed full activity effort |
| $Ax\le e$ | resource capacity cannot be exceeded |
| $A_0x=A_0\beta$ | no-rejection input rows cannot be idled |
| $(Cx)_i=0$ when $z_i=0$ | activities that consume an empty buffer receive zero allocation |

If there are no no-rejection input rows, the equality $A_0x=A_0\beta$ is vacuous. Rejection-admissible input rows remain controlled through $Ax\le e$.

For critical rows in $\mathsf K$, $A\beta=e$, so $Ax\le e$ implies $\mathsf K(\beta-x)\ge0$ on those rows. Appended nonbasic rows impose only $x_j\ge0$, already included above.

### 3.3 Objective

For a Markov allocation rule $u(z)\in\mathcal X(z)$, define the induced deviation rate

$$
\theta_u(z)=b(\beta-u(z)).
$$

Let $Z^u$ be the controlled state under $u$. The modified objective is

$$
V^u(z)=\mathbb E_z\left[\int_0^\infty e^{-\gamma t}\lbrace \tilde h\cdot Z^u(t)+\tilde c\cdot \mathsf K\theta_u(Z^u(t))\rbrace\,dt+\int_0^\infty e^{-\gamma t}\chi\cdot dL(t)\right].
$$

The boundary-cost vector is

$$
\chi=Q^\top\mathsf K^\top\tilde c.
$$

The value function is

$$
V(z)=\inf_u V^u(z).
$$

When $Q$ is chosen so that $\chi=0$, the boundary-cost term in the objective vanishes.

### 3.4 HJB Equation

For a smooth test function $f$, define

$$
\mathcal L f(z)=\frac12\mathrm{tr}\lbrace \Gamma D^2f(z)\rbrace.
$$

Inside the minimization, write

$$
\theta_x=b(\beta-x).
$$

The formal HJB equation is

$$
\begin{aligned}
\mathcal L V(z)+\zeta\cdot\nabla V(z)+\tilde h\cdot z
+\min_{x\in\mathcal X(z)}\left\lbrace \nabla V(z)\cdot R\theta_x+\tilde c\cdot\mathsf K\theta_x\right\rbrace
=\gamma V(z).
\end{aligned}
$$

At boundary points, $\mathcal X(z)$ removes allocations from empty buffers, while $HL$ reflects Brownian noise back into the orthant. The boundary condition on face $\lbrace z_i=0\rbrace$ is

$$
\nabla V(z)\cdot H_{\cdot i}+\chi_i=0.
$$

### 3.5 Policy Index

The HJB minimization is a linear program in $x$. For a gradient vector $g\in\mathbb R^I$, define

$$
\pi_j(z;g)=b\lbrace g\cdot R_{\cdot j}+(\mathsf K^\top\tilde c)_j\rbrace,\qquad j\in\mathcal J.
$$

For the value function, use $\pi_j(z)=\pi_j(z;\nabla V(z))$. Terms involving $\beta$ are constant in the minimization, so the Brownian allocation rule solves

$$
x^{\ast}(z)\in\arg\max_{x\in\mathcal X(z)}\sum_{j\in\mathcal J}\pi_j(z)x_j.
$$

This LP is the general policy-extraction rule. Special network structure may make the LP decompose into smaller block problems; those decompositions are implementation shortcuts, not part of the general formulation.

### 3.6 Example Parameters

Section 2 lists the initial BCP data $(\Gamma,\zeta,R,\tilde h,\tilde c,\mathsf K)$. The modified BCP additionally needs the following rows.

| Row | Role |
|---|---|
| $b$ | finite drift/allocation bound; natural scale is $b=\sqrt n=20$ when $n=400$ |
| $\omega$ | boundary reallocation fraction; these examples use $\omega=0.99$ |
| $Q$ | boundary correction in activity space |
| $H=RQ$ | reflection matrix in state space |
| $\chi=Q^\top\mathsf K^\top\tilde c$ | boundary cost vector |
| $\mathcal B_k(z)$ | row-wise feasible activity sets used by the policy index |

The matrices below write the nonzero service rows $Q_{\mathrm{svc}}$. Under the unified input-as-activity convention of Section 1, append zero rows for input activities, because boundary reflection does not reject or create external input. In all three paper examples, $Q$ is chosen so that $\chi=0$.

For these single-resource, unit-capacity examples, $Q$ is built by equal sharing of released capacity. On face $z_i=0$, define

$$
\mathcal S_i=\lbrace j:C_{ij}>0\rbrace,
\qquad
r_{ki}=\sum_{j\in\mathcal S_i}A_{kj}\beta_j,
$$

and

$$
\mathcal M_{ki}=\lbrace \ell:A_{k\ell}>0,\ C_{i\ell}=0,\ C_{\cdot\ell}\ne0\rbrace.
$$

Set $Q_{ji}=\beta_j$ for $j\in\mathcal S_i$, keep input rows at zero, and for $\ell\in\mathcal M_{ki}$ set

$$
Q_{\ell i}\leftarrow Q_{\ell i}-\frac{\omega r_{ki}}{|\mathcal M_{ki}|}.
$$

The choice $\omega=0.99$ leaves a small amount of slack and makes the listed reflection matrices well posed.

The policy LP also decomposes by resource row in these examples. No-rejection input rows are fixed by $A_0x=A_0\beta$ and are not optimized. For the remaining optimized rows, define

$$
\mathcal B_k(z)=\lbrace j:A_{kj}>0,\ C_{ij}=0\ \text{for every }i\text{ with }z_i=0\rbrace.
$$

For each row $k$, choose

$$
j_k^{\ast}(z)\in\arg\max_{j\in\mathcal B_k(z)}\pi_j(z),
$$

and set

$$
x_j^{\ast}(z)=
\begin{cases}
1, & j=j_k^{\ast}(z)\text{ for a row }k\text{ and }(\iota_k=0\text{ or }\pi_j(z)>0),\\
0, & \text{otherwise}.
\end{cases}
$$

Since costs are nonnegative, $V$ is monotone in the queue state, so $g=\nabla V(z)\ge0$ at smooth points. The adjusted idleness vectors below are written in the resource order of each example.

These adjusted $\iota$ are the *policy-effective* values: they take the model $\iota$ of Section 1.6 and tighten to $\iota_k=0$ any processing row whose block scores $\pi_j(z)$ are provably nonnegative (so the rule $(\iota_k=0\text{ or }\pi_j>0)$ serves it either way). Both choices yield the same policy, so the configuration files store the Section 1.6 model $\iota$ and the code reproduces the adjusted policy automatically.

**Criss-Cross**

| Item | Value |
|---|---|
| service rows in $Q_{\mathrm{svc}}$ | $(s_1,s_2,s_3)$ |
| unified $Q$ | $Q=(0_{2\times3};Q_{\mathrm{svc}})$ for activity order $(a_1,a_2,s_1,s_2,s_3)$ |
| policy blocks | $\mathcal B_{S_1}(z)=\lbrace s_1:z_1>0\rbrace\cup\lbrace s_2:z_2>0\rbrace$, $\mathcal B_{S_2}(z)=\lbrace s_3:z_3>0\rbrace$ |
| boundary cost | $\chi=0$ |

For $g=\nabla V(z)$, the nonzero policy scores under the unified activity order
$(a_1,a_2,s_1,s_2,s_3)$ are

$$
\begin{aligned}
\pi_{a_1}/b&=-g_1,&
\pi_{a_2}/b&=-g_2,\\
\pi_{s_1}/b&=2g_1,&
\pi_{s_2}/b&=2(g_2-g_3),&
\pi_{s_3}/b&=g_3.
\end{aligned}
$$

Use
$$
\iota=(0,0,1,0)
$$
for resource order $(u_1,u_2,S_1,S_2)$.

$$
Q_{\mathrm{svc}}=
\begin{pmatrix}
1/2 & -\omega/2 & 0\\
-\omega/2 & 1/2 & 0\\
0 & 0 & 1
\end{pmatrix},
\qquad
H=
\begin{pmatrix}
1 & -\omega & 0\\
-\omega & 1 & 0\\
\omega & -1 & 1
\end{pmatrix}.
$$

**Pesic-Williams**

| Item | Value |
|---|---|
| service rows in $Q_{\mathrm{svc}}$ | $(s_1,s_2,s_3,s_4,s_5)$ |
| unified $Q$ | $Q=(Q_{\mathrm{svc}};0_{3\times3})$ for activity order $(s_1,\ldots,s_5,a_1,a_2,a_3)$ |
| policy blocks | $\mathcal B_{S_1}(z)=\lbrace s_1:z_1>0\rbrace\cup\lbrace s_5:z_3>0\rbrace$, $\mathcal B_{S_2}(z)=\lbrace s_2:z_1>0\rbrace\cup\lbrace s_3:z_2>0\rbrace$, $\mathcal B_{S_3}(z)=\lbrace s_4:z_3>0\rbrace$ |
| boundary cost | $\chi=0$ |

For $g=\nabla V(z)$, the nonzero policy scores under the unified activity order
$(s_1,\ldots,s_5,a_1,a_2,a_3)$ are

$$
\begin{aligned}
\pi_{s_1}/b&=g_1,&
\pi_{s_2}/b&=2g_1,&
\pi_{s_3}/b&=2g_2,\\
\pi_{s_4}/b&=g_3,&
\pi_{s_5}/b&=g_3,\\
\pi_{a_1}/b&=-2g_1,&
\pi_{a_2}/b&=-g_2,&
\pi_{a_3}/b&=-g_3.
\end{aligned}
$$

Use
$$
\iota=(0,0,0,0,0,0)
$$
for resource order $(S_1,S_2,S_3,u_1,u_2,u_3)$.

$$
Q_{\mathrm{svc}}=
\begin{pmatrix}
1 & 0 & 0\\
1/2 & -\omega/2 & 0\\
-\omega/2 & 1/2 & 0\\
0 & 0 & 1\\
-\omega & 0 & 0
\end{pmatrix},
\qquad
H=
\begin{pmatrix}
2 & -\omega & 0\\
-\omega & 1 & 0\\
-\omega & 0 & 1
\end{pmatrix}.
$$

**Three-Station Bigstep**

| Item | Value |
|---|---|
| service rows in $Q_{\mathrm{svc}}$ | $(s_1,\ldots,s_8)$ |
| unified $Q$ | $Q=(Q_{\mathrm{svc}};0_{4\times8})$ for activity order $(s_1,\ldots,s_8,a_9,a_{10},a_{11},a_{12})$ |
| policy blocks | $\mathcal B_{S_1}(z)=\lbrace s_1:z_1>0\rbrace\cup\lbrace s_7:z_7>0\rbrace$, $\mathcal B_{S_2}(z)=\lbrace s_2:z_2>0\rbrace\cup\lbrace s_5:z_5>0\rbrace$, $\mathcal B_{S_3}(z)=\lbrace s_3:z_3>0,s_4:z_4>0,s_6:z_6>0,s_8:z_8>0\rbrace$, $\mathcal B_{u_A}(z)=\lbrace a_9,a_{10}\rbrace$, $\mathcal B_{u_B}(z)=\lbrace a_{11}\rbrace$, $\mathcal B_{u_C}(z)=\lbrace a_{12}\rbrace$ |
| boundary cost | $\chi=0$ |

Let
$$
c_A=0.165,\qquad c_B=0.100,\qquad c_C=0.225
$$
be the three input components of $\tilde c$. For $g=\nabla V(z)$, the policy scores under the unified activity order
$(s_1,\ldots,s_8,a_9,a_{10},a_{11},a_{12})$ are

$$
\begin{aligned}
\pi_{s_1}/b&=\tfrac12(g_1-g_2),&
\pi_{s_2}/b&=\tfrac13 g_2,&
\pi_{s_3}/b&=g_3,\\
\pi_{s_4}/b&=g_4-g_5,&
\pi_{s_5}/b&=g_5,&
\pi_{s_6}/b&=g_6-g_7,\\
\pi_{s_7}/b&=\tfrac12(g_7-g_8),&
\pi_{s_8}/b&=g_8,\\
\pi_{a_9}/b&=c_A-\tfrac12 g_1,&
\pi_{a_{10}}/b&=c_A-\tfrac12 g_3,\\
\pi_{a_{11}}/b&=c_B-\tfrac14 g_4,&
\pi_{a_{12}}/b&=c_C-\tfrac14 g_6.
\end{aligned}
$$

Use
$$
\iota=(1,0,1,1,1,1)
$$
for resource order $(S_1,S_2,S_3,u_A,u_B,u_C)$. The $S_3$ row stays idle-admissible because, after empty-buffer masking, the feasible set may contain only $s_4$ or $s_6$, whose scores $g_4-g_5$ and $g_6-g_7$ need not be nonnegative.

$$
Q_{\mathrm{svc}}=
\begin{pmatrix}
1/2 & 0 & 0 & 0 & 0 & 0 & -\omega/2 & 0\\
0 & 3/4 & 0 & 0 & -\omega/4 & 0 & 0 & 0\\
0 & 0 & 1/4 & -\omega/12 & 0 & -\omega/12 & 0 & -\omega/12\\
0 & 0 & -\omega/12 & 1/4 & 0 & -\omega/12 & 0 & -\omega/12\\
0 & -3\omega/4 & 0 & 0 & 1/4 & 0 & 0 & 0\\
0 & 0 & -\omega/12 & -\omega/12 & 0 & 1/4 & 0 & -\omega/12\\
-\omega/2 & 0 & 0 & 0 & 0 & 0 & 1/2 & 0\\
0 & 0 & -\omega/12 & -\omega/12 & 0 & -\omega/12 & 0 & 1/4
\end{pmatrix}.
$$

$$
H=
\begin{pmatrix}
1/4 & 0 & 0 & 0 & 0 & 0 & -\omega/4 & 0\\
-1/4 & 1/4 & 0 & 0 & -\omega/12 & 0 & \omega/4 & 0\\
0 & 0 & 1/4 & -\omega/12 & 0 & -\omega/12 & 0 & -\omega/12\\
0 & 0 & -\omega/12 & 1/4 & 0 & -\omega/12 & 0 & -\omega/12\\
0 & -3\omega/4 & \omega/12 & -1/4 & 1/4 & \omega/12 & 0 & \omega/12\\
0 & 0 & -\omega/12 & -\omega/12 & 0 & 1/4 & 0 & -\omega/12\\
-\omega/4 & 0 & \omega/12 & \omega/12 & 0 & -1/4 & 1/4 & \omega/12\\
\omega/4 & 0 & -\omega/12 & -\omega/12 & 0 & -\omega/12 & -1/4 & 1/4
\end{pmatrix}.
$$

## 4. BSDE Training Formulation

The BSDE solver trains $V$ and $\nabla V$ from simulated paths of a fixed reference diffusion. The reference policy is used only for sampling; the HJB Hamiltonian still computes the optimizing allocation $x_g^{\ast}(z)$.

### 4.1 Reference Process

Choose a fixed Markov allocation rule

$$
\bar u(z)\in\mathcal X(z),
$$

and define

$$
\bar\theta(z)=b(\beta-\bar u(z)).
$$

Let $\sigma$ satisfy

$$
\sigma\sigma^\top=\Gamma.
$$

The reference reflected diffusion is

$$
d\bar Z(t)=\sigma\,dW(t)+\zeta\,dt+R\bar\theta(\bar Z(t))\,dt+H\,d\bar L(t),
\qquad \bar Z(t)\in\mathbb R_+^I.
$$

The local-time process $\bar L$ satisfies the same Skorokhod conditions as in Section 3.1.

### 4.2 Hamiltonian Driver

For $g\in\mathbb R^I$, compute the score $\pi_j(z;g)$ from Section 3.5 and solve

$$
x_g^{\ast}(z)\in\arg\max_{x\in\mathcal X(z)}\sum_{j\in\mathcal J}\pi_j(z;g)x_j.
$$

Set

$$
\theta_g^{\ast}(z)=b(\beta-x_g^{\ast}(z)).
$$

The Hamiltonian driver is

$$
F(z,g)=\tilde h\cdot z+g\cdot R\theta_g^{\ast}(z)+\tilde c\cdot\mathsf K\theta_g^{\ast}(z).
$$

The HJB equation is

$$
\mathcal L V(z)+\zeta\cdot\nabla V(z)+F(z,\nabla V(z))=\gamma V(z).
$$

### 4.3 Ito Identity

Apply Ito's formula to $e^{-\gamma t}V(\bar Z(t))$. Using the HJB equation and the boundary condition

$$
\nabla V(z)\cdot H_{\cdot i}+\chi_i=0,
$$

gives the path identity

$$
\begin{aligned}
e^{-\gamma T}V(\bar Z(T))-V(\bar Z(0))
=&\int_0^T e^{-\gamma t}\nabla V(\bar Z(t))^\top\sigma\,dW(t)\\
&+\int_0^T e^{-\gamma t}\lbrace \nabla V(\bar Z(t))\cdot R\bar\theta(\bar Z(t))-F(\bar Z(t),\nabla V(\bar Z(t)))\rbrace\,dt\\
&-\int_0^T e^{-\gamma t}\chi\cdot d\bar L(t).
\end{aligned}
$$

When $\chi=0$, the boundary-cost term drops out.

### 4.4 Training Loss

Let $V_\eta$ be the value network and $G_\phi$ be the gradient network. On a simulated reference path, define the residual

$$
\begin{aligned}
\delta_{\eta,\phi}
=&e^{-\gamma T}V_\eta(\bar Z(T))-V_\eta(\bar Z(0))
-\int_0^T e^{-\gamma t}G_\phi(\bar Z(t))^\top\sigma\,dW(t)\\
&+\int_0^T e^{-\gamma t}\lbrace F(\bar Z(t),G_\phi(\bar Z(t)))-G_\phi(\bar Z(t))\cdot R\bar\theta(\bar Z(t))\rbrace\,dt\\
&+\int_0^T e^{-\gamma t}\chi\cdot d\bar L(t).
\end{aligned}
$$

The BSDE loss is

$$
\mathcal L_{\mathrm{BSDE}}(\eta,\phi)=\mathbb E\left[\delta_{\eta,\phi}^2\right].
$$

For implementation on a grid $0=t_0<\cdots<t_N=T$, replace the stochastic integral by

$$
\sum_{m=0}^{N-1}e^{-\gamma t_m}G_\phi(\bar Z(t_m))^\top\sigma\,\Delta W_m,
$$

and replace the time and boundary integrals by corresponding Riemann sums. After training, use $G_\phi$ in the policy-lifting rule of Section 5.

## 5. Policy Lifting to the Queueing Network

The BCP solution gives a value gradient on diffusion scale. Policy lifting turns that gradient into an executable action for the original queueing network.

### 5.1 State Mapping

At a decision epoch in the scale-$n$ queueing system, observe the queue vector

$$
q\in\mathbb Z_+^I.
$$

Use the diffusion-scaled state

$$
z=\frac{q}{\sqrt n}.
$$

Evaluate the trained gradient network

$$
g=G_\phi(z).
$$

Then compute the same activity scores as in Section 3.5:

$$
\pi_j(q)=b\lbrace g\cdot R_{\cdot j}+(\mathsf K^\top\tilde c)_j\rbrace,\qquad j\in\mathcal J.
$$

Here $b$ is the Brownian control bound used by the trained model. In the main experiments, $b=\sqrt n=20$.

### 5.2 Executable Feasible Set

The executable prelimit action set is

$$
\mathcal A(q)=\lbrace a\in\lbrace 0,1\rbrace^J:Aa\le e,\quad Ca\le q,\quad A_0a=A_0\beta\rbrace.
$$

| Constraint | Meaning |
|---|---|
| $a\in\lbrace 0,1\rbrace^J$ | each executable activity is either selected or not selected |
| $Aa\le e$ | resource capacity is respected |
| $Ca\le q$ | selected service activities cannot consume unavailable jobs |
| $A_0a=A_0\beta$ | no-rejection input streams are always admitted |

If the simulator allows fractional effort over a decision interval, replace $\lbrace 0,1\rbrace^J$ by the appropriate continuous action set. If the model has explicit rejection actions rather than input idleness, include those actions in $\mathcal A(q)$ and charge them using the scale-$n$ rejected-job costs from Section 2.3.

If there are no no-rejection input rows, the equality $A_0a=A_0\beta$ is vacuous.

### 5.3 Lifted Index Policy

The lifted BCP policy chooses

$$
a^{\ast}(q)\in\arg\max_{a\in\mathcal A(q)}\sum_{j\in\mathcal J}\pi_j(q)a_j.
$$

This is the prelimit analog of the Brownian LP in Section 3.5. The only difference is that the action must be executable for the current integer queue state $q$.

### 5.4 Paper Example Simplification

For the three paper examples, the executable optimization decomposes by resource block after no-rejection input rows are fixed.

For a resource row $k$, define the feasible block

$$
\mathcal B_k(q)=\lbrace j:A_{kj}>0,\ C_{\cdot j}\le q\rbrace.
$$

For each optimized row $k$, choose

$$
j_k^{\ast}(q)\in\arg\max_{j\in\mathcal B_k(q)}\pi_j(q).
$$

The row action is

$$
a_j^{\ast}(q)=
\begin{cases}
1, & j=j_k^{\ast}(q)\text{ for a row }k\text{ and }(\iota_k=0\text{ or }\pi_j(q)>0),\\
0, & \text{otherwise}.
\end{cases}
$$

If $\mathcal B_k(q)=\emptyset$, resource $k$ idles when idleness is allowed. If row $k$ is an input row with rejection allowed, choosing no activity in the row corresponds to rejection/no-admission.

For the Three-Station routing input row $u_A$, the block $\mathcal B_{u_A}(q)=\lbrace a_9,a_{10}\rbrace$ selects the larger of the two routing scores when at least one is positive; otherwise type-A input is rejected. The $u_B$ and $u_C$ rows are single-admission blocks and use the same zero-threshold rule.
