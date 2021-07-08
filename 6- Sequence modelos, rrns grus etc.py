#Recurrent neural network
#RNNs sao redes neurais que sao boas em modelar sequence data
#Imagine que voce tira  uma fota de uma bola movendo e voce quer adivinhar a direcao que ela vai somente com a imagen
#Agora imagine que voce tirou um monte de fotos
#O nome disso e data sequencial, ou sequencial
#Outros exemplos, sao audio. Textos tambem sao outras
#RNNs tem um loop dentros dos hidden layers, o nome disso e hidden state
#O problema dela e que ela tem uma memoria curta, devido ao problema de descendencia gradiente diminutiva,
#Imagine o peso inicial de uma funcao, ele esta prestes a ser atualizado pelo processo de backpropagation
#Como a formula indica, novo peso= peso velho -learning rate *gradiente
#Se o gradiente foi muito pequeno. Ela nao vai se atualizar significamente,
#A gente chama esse problema de memoria curta, tendo em consideracao que as infos dos dados anteriores atingiram poucos
#Para combater isso usamos de lstms e grus que seram explicadas a seguir
#LSTMS
#Para combate o problema de descendencia gradiente diminutiva
#LSTMs se utilizam de gates(portoes), para entender qual daquela data e importante manter e qual nao e
#Imagine um comentario numa caixa de cereal, semelhante ao seu cerebro. A rede neural ira selecionar somente as palavras importantes
#De carater avaliativo como bom e tals.
#Para fazer isso a lstm, contem um local chamado status de celula(cell state), ele e tipo uma via rapida
#Que carrega a informacao ate a sequence chain, como uma memoria. 
#Existe 3 tipos de gates, forget gate, input gate e output gate. O forget gate, decide qual informacao e jogada fora e qual nao e
#Ele pegas as infromacoes do hidden state antigo e o input atual e passa ela pela sigmoid function (funcao entre 1 e 0)
#Valores entre 0 e 1 saiem fora, valores perto de 0 saiem, perto de 1 ficam
#O input gate, semelhante ao forget gate tambem pega o hidden state antigo e o imput atual
#No forget gate existe passos,
#Primeiro a gente passa, esses dois valores pela sigmoid function e decide qual dos valores irao ser atualizados, 0 nao e importante 1 e
#Depois voce passa por uma tahn function(funcao entre 1 e -1), regulando a network
#Depois voce multiplica o tahn output pelo sigmoid output decidindo qual informacao e importante manter
#Depois a gente pega o output do forget gate, e multiplica pelo cell gate
#Isso tem a habilidade de dropar valores
#Depois a gente pega o output candidadot(forget gate por cell gate)
#E multiplica pelo o output candidato(do input gate)
#Isso da o nosso novo cell state 
#Depois a gente pega o hidden state e o input e passa por outra sigmoid 
#Depois a gente pego o novo cell state e passa pela sigmoid function
#A gente multiplica esse dois, e a gente consegue o hidden state
#Depois agente carrega o novo cell gate e hidden state para o proximo step
#GRU
#Gru e muito semlhante a lstm
#GRU se livraram do cell state e usam um hidden state para transformar informacao
#Ela tambem tem dois gates, a recicle gate e um update gate
#Um update gate e um combo de input gate com forget gate
#Decide qual informacao jogar fora e qual adicionar
#O reset gate decide quais informacoes que passaram esquecer
#RNNS BIDIDIRECIONAIS
#Imagine que uma das palavras que voce tem, precisa de uma palavra que ira ser analisada futaramente em outro node
#Para evitar isso nos se utilizamos de rnn bidirecionais
#Agora imagine que temos dois flows de tempo
#1 flow para frente com certos nodes
#1 flow para tras com certos nodes
#Pensa que o flow iria da origem do input ate o output final (funcao final)
#Agora pensa que o output e os inputs dos nodes sao combinados, inputs antes de entrar no node, e outputs depois
#Agora nos temos os inputs combados, pq o input x1,4 tambem sera associado ao input da x1,3
#E bem lenta no entanto, cuidado.