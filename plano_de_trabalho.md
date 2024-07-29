# Questões
Competir no tempo do DFS seria o ideal.

Quando o DFS encontra um nó inviável, ele automaticamente checa o vizinho.

O FBS, por outro lado, retoma o trecho do começo.

É possível definir de modo razoavelmente justo instâncias com espaçamento (j - i) regular?

Quando contamos as frequências dos binários, assumimos que está é a única solução para o trecho observado, mas isto pode não ser verdade. Qual é a a veracidade desta hipótese? Ou seja, as restrições podem ter muitas soluções alternativas. Quantas soluções existem para uma dada restrição?

Talvez não seja possível bater o tempo do DFS, utilizando a FBS. Uma implementação em C, pode não ser suficiente.

Número de equações não-lineares resolvidas é uma métrica necessária, ainda que não seja suficiente pois estamos ignorando o overhead das demais estruturas. Portanto, podemos continuar a implementação em Python até obtermos vantagem nesta estrutura.

# Instâncias
DMDGP vs DDGP
- DMDGP é muito artificial e não reflete as hipóteses que temos.
- DDGP é mais realista e cada resíduo contém apenas um átomo que não satisfaz as hipóteses do DMDGP (nitrogênio, (i-1,i-2,i-4)).

Orderm DDGP
(N1, 0), (CA1, 1), (HA1, 2), (C1, 3), (H2,4), (N2,5), (CA2, 6) (HA2, 7), (C2, 8), ...

Reordem para utilizarmos uma formulação DMDGP
N1 HA1 C1 CA1 H2 N2 CA2 HA2 C2 CA2 H3 N3 CA3 HA3 C3 CA3 H4 N4 CA4 HA4 C4 CA4 H5 N5 CA5 HA5 C5 CA5  
0  1   2  3   4  5  6   7   8  9'  10 11 12  13  14 15' 16 17 18  19  20 21' 22 23 24  25  26 27'
          *         *                    *                    *                    *

H-6-H
H-9-HA
H-12-H

HA-6-HA
HA-9-H
HA-12-HA

Esta reordem mantém a ordem de fixação da DDGP original.

DFS
ai----------------bi-----aj------bj

(25, 33, 5.3)
b[28:33] = 010101010
