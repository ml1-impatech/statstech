# Decisões

Este documento registra as decisões tomadas na construção da biblioteca `statstech`.

## Decisões gerais

### 1. Organização do repositório

- Inspirada na biblioteca [CB2325NumericaG8](https://github.com/lazyluca/CB2325NumericaG8),
  construída por alunos do IMPA Tech na disciplina de Programação 2 (2025).
- **`src/statstech/`**: o código, uma pasta por módulo (`linear_model`, `tree`,
  `neural_network`, `metrics` etc.). O `__init__.py` de cada pasta define o que
  é público, para o uso ser `from statstech.linear_model import Ridge`.
- **`tests/`**: a mesma estrutura de pastas de `src/`, com um arquivo de teste por arquivo de código.
- **`examples/`**: um exemplo simples de uso para cada módulo.
- **`docs/`**: este documento.

### 2. Classe ou função

- **Classe** quando o objeto guarda estado aprendido: modelos e transformadores.
- **Função** quando só recebe entrada e devolve saída, sem estado: métricas,
  funções de validação e de reamostragem.

### 3. Código compartilhado

Funcionalidades usadas por vários módulos ficam num lugar só, para evitar repetição de código.

- **`base.py`**: classe base dos modelos (`BaseEstimator`), os mixins
  (`ClassifierMixin`, `RegressorMixin`, `TransformerMixin`) e `clone`.
- **`utils/validation.py`**: todos os modelos validam as entradas do mesmo jeito e
  levantam as mesmas mensagens de erro (`check_X_y`, `check_array`,
  `check_is_fitted`, `check_n_features`).
- **`utils/distances.py`**: `pairwise_distances`, usada pelo KNN e pelo K-means.

### 4. Contrato dos modelos

- O `__init__` só guarda os hiperparâmetros, cada um num atributo de mesmo nome.
  É o que permite `get_params`, `set_params` e `clone`.
- A validação dos hiperparâmetros fica em `_check_params()`, chamado no início
  do `fit`. Assim, um valor passado por `set_params` também é validado.
- Atributos aprendidos no `fit` terminam com `_` (`coef_`, `classes_`).
- Todo `predict` e `transform` confere se X tem o mesmo número de features do `fit`.

### 5. Pré-processamento

- Os modelos não padronizam os dados por dentro. Quem usa padroniza com o
  `StandardScaler` do módulo `preprocessing`.
- Motivos: evita repetir o mesmo código em cada modelo e mantém cada módulo com
  uma responsabilidade.
- Cuidado na validação cruzada: o `StandardScaler` precisa ser ajustado só com
  os dados de treino de cada fold. Padronizar a base inteira antes do
  `cross_val_score` faz a média do fold de validação vazar para o treino.

### 6. Reprodutibilidade

- Tudo que sorteia (divisões, folds, bootstrap, inicialização do K-means e da
  rede) recebe `random_state` e usa `np.random.default_rng(random_state)`.
  Com a mesma semente, o resultado é o mesmo.

### 7. Estilo de código

- **PEP 8**, verificado com `ruff check` e `ruff format`.
- **Código em inglês** (classes, funções, variáveis e parâmetros) e **docstrings
  e comentários em português**.
- **Docstrings no padrão NumPy**, com as seções `Parameters`, `Attributes`,
  `Returns`, `Raises` e, quando necessário, `Notes`. Não usamos `Examples` nem
  `References`.
- **Type hints** nos parâmetros e no retorno de todas as funções e métodos.
- **Imports absolutos**: `from statstech.base import BaseEstimator`.

### 8. Fluxo de trabalho no Git

- A `main` é **protegida**: mudanças só por Pull Request, com revisão.
- Uma branch por tarefa.
- Problemas e propostas de mudança são registrados como *issues*.

### 9. Instalação

- Requer Python 3.11 ou mais recente e NumPy 1.26 ou mais recente.
- Instalável por `pip install git+https://github.com/ml1-impatech/statstech.git`.

## Decisões de cada método

### 10. Modelos lineares

- **`LinearRegression` e `Ridge`** resolvem por `np.linalg.lstsq`, que funciona
  mesmo com colunas colineares (por exemplo, one-hot sem `drop="first"`).
- **Intercepto:** com `fit_intercept=True`, X e y são centralizados, e o
  intercepto não é penalizado no `Ridge` nem no `Lasso`.
- **`Lasso`:** minimiza (1/2n)‖y − Xβ‖² + α‖β‖₁, a mesma escala do livro texto
  (ISL) e do scikit-learn, por descida coordenada. Com
  α ≥ max|Xᵀy|/n (X e y centralizados), todos os coeficientes são zero.
- **`LogisticRegression`:** multiclasse com softmax, treinada por gradiente
  descendente, com penalidade L2 (α/2n)‖β‖² que não inclui o intercepto. A perda
  de cada iteração fica em `loss_history_`.

### 11. Vizinhos e árvores

- **KNN:** distância euclidiana. Empate de distância: vence o ponto que aparece
  primeiro no treino. Empate no voto: vence a menor classe em `classes_`.
- **Árvores:** critério de Gini (classificação) e variância (regressão). O corte
  é o ponto médio entre dois valores distintos consecutivos. Empate entre
  cortes: vence o primeiro encontrado.

### 12. Análise discriminante (LDA e QDA)

- **Regularização:** os dois usam `reg_param=1e-6` como padrão, somado à
  diagonal das matrizes de covariância.
- **Validação de `reg_param`:** os dois rejeitam valores negativos, não finitos
  e booleanos.
- **Priors:** os dois aceitam o parâmetro `priors`. Com `priors=None`, as
  probabilidades a priori são as frequências das classes no treino.

### 13. Rede neural

- `MLPRegressor` com saída linear, perda MSE, ativação ReLU nas camadas ocultas,
  inicialização He e gradiente descendente em mini-lotes.

### 14. Não supervisionado e pré-processamento

- **PCA:** pela SVD do X centralizado, com variância s²/(n − 1). Convenção de
  sinal: a maior entrada em módulo de cada componente é positiva.
- **K-means:** inicialização k-means++ por padrão. Um cluster vazio recebe o
  ponto mais distante do seu centro. Para quando o deslocamento dos centros
  fica menor ou igual a `tol`.
- **`StandardScaler`:** desvio-padrão com divisão por n. Uma coluna constante é
  dividida por 1, em vez de gerar NaN.

### 15. Validação, busca e bootstrap

- **Divisores:** `KFold`, `StratifiedKFold` e `LeaveOneOut` têm a mesma
  interface, `split(X, y=None)`. O `StratifiedKFold` usa o `y` para manter
  em cada fold a proporção de cada classe; os outros dois o ignoram. Assim,
  `cross_val_score`, `cross_val_predict` e `GridSearchCV` aceitam qualquer um
  deles como `cv`.
- **`cv` inteiro:** usa `KFold(cv)` sem embaralhar.
  Recomendamos passar o divisor explícito, com `shuffle=True` e
  `random_state` inteiro. O `random_state` inteiro garante os mesmos folds a
  cada chamada, e portanto que todas as combinações de uma busca são
  avaliadas nas mesmas partições.
- **`StratifiedKFold`:** cada classe precisa ter pelo menos `n_splits`
  exemplos; senão algum fold de teste ficaria sem ela, e a função levanta
  erro.
- **`cross_val_predict`:** prevê cada amostra com um modelo que não a viu,
  e levanta erro se o `cv` não colocar cada amostra no teste exatamente uma
  vez. Com `method="predict_proba"`, as colunas seguem `np.unique(y)`; se um
  fold de treino não tiver alguma classe, a probabilidade dela é 0 nas
  amostras daquele fold.
- **`GridSearchCV` e métricas por nome:** com um nome de métrica de erro
  (`"mse"`, `"rmse"`, `"mae"`, `"log_loss"`), a busca escolhe o menor valor;
  nos demais casos, o maior. Uma função passada diretamente é sempre
  maximizada, então métricas de erro devem ser passadas pelo nome.
- **`error_score`:** com o padrão `np.nan`, uma combinação que dá erro
  recebe `NaN`, emite um aviso e nunca é escolhida; a busca continua. Com
  `error_score="raise"`, o erro interrompe a busca.
- **`cv_results_`:** guarda, para cada combinação, a média, o desvio-padrão
  amostral (`ddof=1`) e as notas de cada fold. O desvio permite dizer se a
  diferença entre duas combinações é maior que a variação entre folds.
- **`bootstrap` e `bootstrap_ci`:** o `bootstrap` devolve as estimativas
  reamostradas e o erro-padrão. O intervalo de confiança fica numa função
  separada, `bootstrap_ci`, para não mudar o que o `bootstrap` devolve. O
  intervalo é calculado pelo método do percentil; se a estatística devolve
  um vetor, cada posição recebe o seu intervalo.
- **Precisão, revocação, especificidade e F1** com denominador zero devolvem 0
  e emitem um `UserWarning`.
- **R² com y constante:** devolve 1 se a previsão for perfeita e 0 caso contrário.