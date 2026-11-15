# Motor matemático de álgebra linear

Um programa que resolve exercícios de álgebra linear e mostra a resolução completa, passo a passo, como num caderno.

Você escreve o exercício num arquivo de texto, roda um comando e recebe a resposta, as contas intermediárias e uma conferência de que o resultado está certo. As respostas são exatas: em vez de `0.4444...`, o programa mostra `4/9`; em vez de `14.3178...`, mostra `raiz(205)`.

Projeto da disciplina de Álgebra Linear Aplicada, Profa. Catiuscia Albuquerque.

---

## Passo 1: instalar o Python

O programa precisa do **Python 3.11 ou mais novo**.

Para saber se você já tem, abra o terminal (veja abaixo como) e digite:

```
python3 --version
```

No Windows, digite `python --version`.

Se aparecer algo como `Python 3.12.1`, está tudo certo (o número do meio precisa ser 11 ou maior). Se aparecer um erro ou uma versão antiga, baixe o Python em **python.org/downloads** e instale. No Windows, durante a instalação, **marque a opção "Add Python to PATH"**.

### Como abrir o terminal

- **Mac:** aperte Cmd + Espaço, digite `Terminal` e aperte Enter.
- **Windows:** clique no menu Iniciar, digite `PowerShell` e aperte Enter.

O terminal é uma janela onde você digita comandos e aperta Enter para executá-los.

---

## Passo 2: baixar o programa

1. Nesta página do GitHub, clique no botão verde **Code** e depois em **Download ZIP**.
2. Descompacte o arquivo baixado (no Mac, dois cliques; no Windows, botão direito e "Extrair tudo").
3. Você terá uma pasta chamada `motor-algebra-linear-main`. Mova para onde preferir, como a Área de Trabalho.

---

## Passo 3: entrar na pasta pelo terminal

O terminal precisa estar "dentro" da pasta do programa. Digite `cd ` (com um espaço no final), **arraste a pasta para dentro da janela do terminal** e aperte Enter.

Vai ficar parecido com isto:

```
cd /Users/seunome/Desktop/motor-algebra-linear-main
```

---

## Passo 4: testar com o exemplo pronto

O programa já vem com 8 exercícios resolvidos. Para ver a resolução completa, digite:

```
python3 -m lae.cli explicar exemplos/aps.lae
```

No Windows, troque `python3` por `python` em todos os comandos deste guia.

Se aparecerem os exercícios resolvidos, terminando com uma linha **Resumo: ... 0 falha(s)**, está tudo funcionando.

Existem dois modos:

| Comando | O que mostra |
|---|---|
| `python3 -m lae.cli explicar arquivo.lae` | A resolução completa, passo a passo |
| `python3 -m lae.cli resolver arquivo.lae` | Só as respostas finais |

---

## Passo 5: resolver os seus exercícios

### Criando o arquivo

1. Abra um editor de texto simples:
   - **Mac:** TextEdit. Antes de escrever, clique em **Formatar → Converter em Texto Simples**.
   - **Windows:** Bloco de Notas.
2. Escreva o exercício (veja como logo abaixo).
3. Salve **dentro da pasta do programa** com um nome terminado em `.lae`, como `lista1.lae`.
   - No Bloco de Notas, em "Tipo", escolha **Todos os arquivos** para ele não acrescentar `.txt` no final.
4. No terminal, rode:

```
python3 -m lae.cli explicar lista1.lae
```

### Traduzindo o enunciado

O programa não lê o enunciado em português. Você "traduz" o exercício para uma escrita simples, que é quase igual à do caderno.

**Exemplo.** O enunciado:

> Sejam u = (3, 0, 1) e v = (-5, 1, 2). Calcule 2u + 3v.

vira:

```
u = (3, 0, 1)
v = (-5, 1, 2)
2u + 3v
```

Ou seja: primeiro você informa os dados, depois escreve o que quer calcular. Cada coisa numa linha.

### Como escrever cada coisa

| O que é | Como escrever | Exemplo |
|---|---|---|
| Vetor | números entre parênteses, separados por vírgula | `(3, 0, -1)` |
| Matriz | cada linha entre colchetes, tudo entre colchetes | `[[3, 7], [8, 4]]` |
| Fração | com barra | `1/3` |
| Decimal | com **ponto**, não vírgula | `2.5` |
| Dar nome a algo | nome, sinal de igual, valor | `u = (3, 0, 1)` |
| Soma e subtração | `+` e `-` | `u + v` |
| Multiplicação | `*`, ou o número colado no nome | `2*u` ou `2u` |
| Título | linha começando com `##` | `## Exercício 1` |
| Anotação sua | linha começando com `#` (o programa ignora) | `# lista da semana 3` |

A matriz `[[3, 7], [8, 4]]` é esta:

```
| 3  7 |
| 8  4 |
```

### O que dá para calcular

| Para calcular... | Escreva | Exemplo |
|---|---|---|
| Distância entre dois vetores | `dist(u, v)` | `dist((1, 2), (4, 6))` |
| Cosseno do ângulo entre vetores | `cos(u, v)` | `cos(u, v)` |
| Norma (tamanho) de um vetor | `norma(v)` | `norma((3, 4))` |
| Produto escalar | `dot(u, v)` | `dot(u, v)` |
| Determinante | `det(A)` | `det([[1, 2], [3, 4]])` |
| Matriz transposta | `transposta(A)` | `det(transposta(A))` |
| Posto de uma matriz | `posto(A)` | `posto(A)` |
| Escalonamento | `escalonar(A)` | `escalonar(A)` |
| Se vetores são LI ou LD | `independente(u, v, w)` | `independente((1, 0), (2, 0))` |
| Coordenadas de w na base {u, v} | `coordenadas(w, u, v)` | `coordenadas(w, u, v)` |

### Exercícios com incógnitas (sistemas)

Escreva `resolver`, os nomes das incógnitas, dois-pontos e a igualdade:

> Determine x e y tais que (-6, 7) = x(3, 4) + y(2, 3).

```
resolver x, y: (-6, 7) = x*(3, 4) + y*(2, 3)
```

Atenção: aqui o `*` é obrigatório. Escreva `x*(3, 4)`, e não `x(3, 4)`.

Depois de resolver, os valores ficam guardados. Se o exercício pede x + y, escreva na linha de baixo:

```
x + y
```

### Subespaços

> Verifique se H = {(x, 12x, y) | x, y ∈ R} é subespaço.

```
subespaco {(x, 12x, y) | x, y}
```

Quando o conjunto é dado por uma equação:

> W = {(x, y, z) | x + y - z = 0}

```
subespaco {(x, y, z) | x + y - z = 0}
```

O programa responde se é subespaço e, se for, a dimensão e uma base. Escreva `subespaco` **sem cedilha**.

### Transformações lineares

> Seja T(x, y, z) = (x + z, y + z). Calcule T(1, 2, 10).

```
T(x, y, z) = (x + z, y + z)
T(1, 2, 10)
```

Também dá para pedir o núcleo e a imagem: `nucleo(T)` e `imagem(T)` (sem acento).

### Exemplo de arquivo completo

```
## Exercício 1
u = (3, 0, 1)
v = (-5, 1, 2)
w = (0, 1, 4)
2u + 3v - 5w

## Exercício 2
A = [[-2, 4, 2], [-2, 3, 1], [-1, 2, 0]]
det(transposta(A))
```

Abra o arquivo `exemplos/aps.lae` para ver mais 8 exercícios escritos assim.

---

## Entendendo a resposta

No modo `explicar`, cada exercício aparece assim:

- **Dado:** os valores que você informou.
- **>>** a conta que você pediu, seguida das contas intermediárias.
- **Resultado:** a resposta. Quando ela tem raiz, aparece também o valor aproximado, como `raiz(205)  (aprox. 14.3178)`.
- **Verificação:** o programa confere a resposta por outro caminho. `[ok]` significa que a conferência bateu.

No fim aparece um **Resumo**. Se ele mostrar `0 erro(s)` e `0 falha(s)`, tudo foi resolvido e conferido.

O programa escreve `raiz(2)` para √2 e `x^2` para x².

---

## Quando algo dá errado

| Mensagem | O que fazer |
|---|---|
| `command not found: python` | No Mac, use `python3` em vez de `python`. |
| `No module named lae` | O terminal não está dentro da pasta do programa. Refaça o Passo 3. |
| `não foi possível ler 'arquivo.lae'` | O nome do arquivo está errado, ou ele não está na pasta do programa. |
| `ERRO DE SINTAXE linha 5: ...` | A linha 5 do seu arquivo foi escrita de um jeito que o programa não entende. A mensagem diz o que está errado. Nada é resolvido até corrigir. |
| `ERRO linha 5: ...` | A conta da linha 5 não é possível, por exemplo somar vetores de tamanhos diferentes. As outras linhas são resolvidas normalmente. |
| `AVISOS` no começo da resposta | O arquivo tinha frases que parecem instruções escondidas (do tipo "ignore as instruções"). Elas foram ignoradas e **não mudam nenhuma resposta**. |

Erros comuns de escrita:

- Vírgula no decimal: escreva `2.5`, não `2,5`.
- Acento ou cedilha nos comandos: escreva `subespaco` e `nucleo`.
- Esquecer o `*` no resolver: escreva `x*(3, 4)`.
- Texto do enunciado solto no arquivo: se quiser guardar o enunciado, comece a linha com `#`.

---

## Para quem quer rodar os testes

Isto é opcional e serve para conferir que o programa está íntegro:

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

No Windows, a segunda linha é `.venv\Scripts\Activate.ps1`. O resultado esperado é `186 passed`.