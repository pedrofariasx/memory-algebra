# Formalização da Álgebra da Memória (Pilar 1)

Este documento fixa os axiomas da álgebra de memórias implementada em `memory_algebra/` e
estabelece a correspondência entre a teoria e o código. A hipótese central é que
uma memória é um objeto composto $(V, G, T, P)$ e que as operações $\oplus$
(compose), $\ominus$ (retract), $\otimes$ (transform) e $\rho$ (retrieve) devem
satisfazer simultaneamente:

- $\boxed{A}$ Associatividade: $(m_1 \oplus m_2) \oplus m_3 \cong m_1 \oplus (m_2 \oplus m_3)$
- $\boxed{R}$ Recuperabilidade: $d_s(m, \rho(M, q)) \to 0$ quando $q$ consulta $m \subseteq M$
- $\boxed{S}$ Estabilidade: $\|M_t\| < C$ sob poda $\Pi_C$
- $\boxed{P}$ Preservação semântica: $\otimes$ não destrói a estrutura recuperável
- $\boxed{E}$ Eficiência: operações tratáveis sobre representações discretas

---

## 1. Definição dos componentes

- $V$: espaço vetorial sobre $\mathbb{R}$ (categoria $\mathbf{Vect}_{\mathbb{R}}$);
  na implementação, $V = \mathbb{R}^d$ com $d$ fixo (`memory_algebra/core.py`, campo `vectors`).
- $G$: grafo rotulado cujos nós carregam vetores de $V$
  (categoria $\mathbf{Grph}$ sobre $\mathbf{Vect}$); implementação em `edges`.
- $T$: monoide ordenado $(\mathbb{R}_{\geq 0}, \leq, +)$; implementação em `times`.
- $P$: semiring idempotente $([0,1], \max, \cdot, 0, 1)$ para utilidade;
  implementação em `weights`.

**Objeto de memória.** Uma memória é a tupla
$m = (N, v, E, \tau, w)$ onde $N$ é um conjunto finito de nós atômicos,
$v : N \to V$, $E \subseteq N \times N \times L$ (rótulos $L$),
$\tau : N \to T$, $w : N \to P$. Implementação: `MemoryObject`
(`memory_algebra/core.py`). A memória **não** é o produto cartesiano $V \times G \times T \times P$,
pois $G$ referencia nós cujos atributos vivem em $V$, $T$ e $P$ — a dependência é
capturada pela estrutura de tuplas indexadas por $N$.

---

## 2. Composição como pushout e prova de associatividade

### 2.1 Construção categórica

A composição $m_1 \oplus m_2$ é o pushout na categoria de grafos tipados:

$$m_1 \oplus m_2 \;=\; m_1 \sqcup_{m_{12}} m_2$$

onde $m_{12} = m_1 \cap m_2$ é a **interseção semântica**: o subobjeto formado
pelos nós de $m_1$ e $m_2$ cujas imagens em $V$ são $\theta$-similares
($\cos(v_a, v_b) \geq \theta$). O pushout identifica os nós correspondentes e
preserva o restante da estrutura disjunta.

### 2.2 Realização computacional exata

O pushout ingênuo (fundir nós por média de embeddings no momento da composição)
**não** é associativo: a média depende da ordem das fusões. A implementação
resolve isso mantendo a estrutura atômica e derivando o quociente sob demanda:

1. $m_1 \oplus m_2$ é a **união disjunta** das tuplas atômicas (`compose`,
   `memory_algebra/core.py`). Nenhuma informação é perdida ou agregada neste passo.
2. A relação de fusão $R_\theta = \{(a,b) : \cos(v_a, v_b) \geq \theta\}$ é
   definida apenas sobre nós atômicos. A partição de fusão é
   $\pi = R_\theta^{*}$ (fecho transitivo), computada por union-find em
   `quotient` (`memory_algebra/core.py`).

### 2.3 Prova de associatividade ($\boxed{A}$)

**Teorema.** $(m_1 \oplus m_2) \oplus m_3 = m_1 \oplus (m_2 \oplus m_3)$
(como objetos atômicos) e os quocientes induzidos são idênticos.

**Prova.**
(i) Como $\oplus$ é união disjunta sobre conjuntos de nós com identificadores
unicamente prefixados, $(N_1 \sqcup N_2) \sqcup N_3 = N_1 \sqcup (N_2 \sqcup N_3)$
pela associatividade de $\sqcup$ em $\mathbf{Set}$; idem para $v$, $E$, $\tau$,
$w$, que são uniões de funções com domínios disjuntos.
(ii) A relação $R_\theta$ sobre o conjunto total $N = N_1 \sqcup N_2 \sqcup N_3$
depende apenas dos pares de vetores atômicos, logo é idêntica em ambas as
associações. O fecho transitivo $R_\theta^{*}$ é único (menor relação de
equivalência contendo $R_\theta$), portanto $\pi$ coincide.
(iii) Embeddings de cluster (média dos membros), tempos ($\max$), pesos ($\max$)
e arestas do quociente são funções determinísticas de $(N, v, E, \tau, w, \pi)$,
logo coincidem. $\blacksquare$

A implementação verifica este teorema por assinaturas canônicas
(`signature`, `memory_algebra/core.py`) e por distância de recuperação nula nos testes
`tests/test_associativity.py` (protocolo 3.1, $N = 100$) e por testes de
propriedade (Hypothesis).

**Corolário (conflitos).** A resolução de conflitos temporais não é feita por
fusão/destruição no pushout, mas pelo operador temporal em $\rho$ (fator
$e^{-((t_c - t_q)/\sigma)^2}$), preservando o histórico ($\boxed{R}$ sem perda).

---

## 3. Lentes delta assimétricas

A lente $\mathcal{L}: \mathcal{M} \rightleftharpoons \mathcal{Q}$ entre o espaço
de estados $\mathcal{M}$ e as vistas/consultas $\mathcal{Q}$ é definida por:

- **Get.** $get(M, q) = \rho(M, q)$ — o subobjeto retornado é a união de
  classes inteiras de $\pi$ (fechado sob fusão), garantindo consistência.
- **Put.** $put(M, q, \Delta) = (M \ominus get(M, q)) \oplus \Delta$
  (`lens_put`, `memory_algebra/lens.py`): retrai a vista corrente e compõe o delta.

**Lei Put-Get.** $get(put(M, q, \Delta), q_\Delta) \cong \Delta$ quando o
embedding de $\Delta$ é distinguível da vizinhança de $q$ e $\Delta$ carrega
tempo/peso que o tornam ótimo para $q_\Delta$. Verificado em
`tests/test_lens.py::test_put_get`.

**Lei Get-Put (vistas isoladas).** $put(M, q, get(M, q)) = M$ vale exatamente
quando a vista é uma união fechada de clusters sem arestas cruzadas para o
complemento (vista isolada), pois a retração remove exatamente os nós da vista
e a recomposição os restaura com identificadores, vetores, tempos e pesos
originais. Verificado em `tests/test_lens.py::test_get_put_isolated_view`.
Para vistas com fronteira aberta, a lei vale na forma fraca
$get(put(M,q,get(M,q)), q) \cong get(M, q)$.

**Functorialidade de $\otimes$ (preservação $\boxed{P}$).** Para $W$ ortogonal,
o quadrado
$\rho \circ \otimes_W = \otimes_W \circ \rho$
comuta: scores de cosseno são invariantes ortogonais e o desempate é por
identificadores, portanto a seleção é idêntica. Verificado em
`tests/test_lens.py::test_transform_functorial`.

---

## 4. Métrica de distância semântica $d_s$

$$d_s(m_1, m_2) = \alpha\, d_V + \beta\, d_G + \gamma\, d_T + \delta\, d_P, \quad \alpha+\beta+\gamma+\delta = 1$$

- $d_V = 1 - \cos(\bar v_1, \bar v_2)$: distância cosseno entre vetores
  agregados (média dos nós) — implementação `semantic_distance`, `memory_algebra/metric.py`.
- $d_G = \tfrac{1}{2}\bigl(1 - J(N_1, N_2)\bigr) + \tfrac{1}{2}\bigl(1 - J(E_1, E_2)\bigr)$:
  distância estrutural por Jaccard sobre nós e arestas (proxy tratável da
  distância de edição de grafos; para grafos grandes, substituir por
  Wasserstein sobre embeddings de grafos).
- $d_T = \frac{|\bar\tau_1 - \bar\tau_2|}{1 + |\bar\tau_1 - \bar\tau_2|}$.
- $d_P = |\bar w_1 - \bar w_2|$.

Pesos padrão $(\alpha, \beta, \gamma, \delta) = (0.4, 0.3, 0.15, 0.15)$.

Os pesos podem ser ajustados dinamicamente pelo contexto da consulta
(`context_weights`, `memory_algebra/metric.py`): em consultas ancoradas no tempo, $\alpha$
transfere até $0.15$ para $\gamma$; em consultas com foco estrutural, $\alpha$
transfere até $0.15$ para $\beta$. A partição $\sum = 1$ é preservada.

O protocolo 3.2 inclui três baselines para comparação com a álgebra: soma
vetorial normalizada, **LSTM** (célula vanilla em `memory_algebra/baselines.py`,
hidden=dim, $h_0 = c_0 = 0$, alimentada pelo vetor unitário do primeiro nó de
cada memória) e buffer FIFO de capacidade 10. A curva do LSTM é computada por
$1 - \cos(h_t, v_1)$, onde $h_t$ é o estado oculto após processar as $t$
memórias sequencialmente.

---

## 5. Estabilidade e poda $\Pi_C$ ($\boxed{S}$)

Definimos $\Pi_C(M)$ (`prune`, `memory_algebra/stability.py`) como a composição de:

1. **Esquecimento por utilidade:** remove $n$ com $w(n) < \varepsilon$.
2. **Compressão de caminhos:** para $u \to b \to v$ com $b$ removido, insere o
   atalho $u \to v$ com rótulo composto, preservando a conectividade relacional
   (transitividade de $G$).
3. **Teto de capacidade:** se $|N| > C$, remove os nós de menor $w$ até $|N| = C$.

**Limitação.** Sob política de poda aplicada após cada composição com
$w \sim$ utilidade estacionária, $|N_t|$ permanece limitado em expectativa por
$C$; memórias com $w = 1$ (âncoras) nunca são removidas pelo esquecimento por
utilidade, e sua recuperabilidade é preservada porque $\oplus$ não altera nós
existentes (prova de estabilidade, protocolo 3.2).

---

## 6. Recuperação $\rho$ ($\boxed{R}$, $\boxed{E}$)

$\rho(M, q)$ (`retrieve`, `memory_algebra/core.py`):

1. Computa $\pi = R_\theta^{*}$ e o grafo quociente $M/\pi$.
2. Score de cada cluster $c$: $s(c) = \max(\cos(q_v, \bar v_c), 0) \cdot f_T(c) \cdot (0.5 + 0.5\, w_c)$,
   com $f_T(c) = e^{-((t_c - t_q)/\sigma)^2}$ se a consulta ancora tempo.
3. Seleciona os $k$ clusters de maior score (foco) e expande $h$ saltos no
   quociente; retorna o subobjeto atômico correspondente.

**Recuperabilidade.** Se $m \subseteq M$ tem peso e ancoragem temporal
compatíveis com $q$ e nenhuma interferência de mesmo suporte, então
$\rho(M, q) = m$ e $d_s(m, \rho(M,q)) = 0$ (verificado nos protocolos 3.1 e 3.2).

**Eficiência.** $\oplus$ é $O(1)$ amortizado (união de estruturas imutáveis);
$\rho$ é $O(|N|^2)$ no quociente (matriz de similaridade) — substituível por
índices ANN para escala, sem alterar a álgebra.

---

## 7. Correspondência teoria-implementação

| Axioma / conceito                  | Implementação                                        |
| :---------------------------------- | :--------------------------------------------------- |
| Objeto $(V, G, T, P)$               | `MemoryObject` (`memory_algebra/core.py`)                     |
| Pushout $m_1 \sqcup_{m_{12}} m_2$   | `compose` + `quotient` (fecho transitivo de $R_\theta$) |
| Associatividade                     | `tests/test_associativity.py`                        |
| Retração $\ominus$ + projeção       | `retract` (`memory_algebra/core.py`)                          |
| Transformação $\otimes$             | `transform` (`memory_algebra/core.py`)                        |
| Recuperação $\rho$                  | `retrieve` (`memory_algebra/core.py`)                         |
| Lentes get/put                      | `memory_algebra/lens.py`, `tests/test_lens.py`                |
| Distância $d_s$                     | `semantic_distance` (`memory_algebra/metric.py`)              |
| Poda $\Pi_C$                        | `prune` (`memory_algebra/stability.py`)                       |
| Protocolos 3.1–3.4                  | `tests/`, `experiments/run_validation.py`            |
