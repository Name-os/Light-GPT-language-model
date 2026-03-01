from settings import *
from resources import Block
from data_handler import decode

class GPTLanguageModel(nn.Module):
    """
    Class Information
    -----------------
    This is the main model, this is what makes predictions based on given text using pretrained

    weights as well as aid from self-attention heads

    Importiant Information
    ----------------------
    -   This model is different from a regular NerualNetwork, it doesn't have a neual network 
    
         but instead giant look up tables and some Linear layers.

    -   One of it's lookup tables is positional embedding where each token at position `i` gets

         a vector added to it at the `i` position on the lookup table that way more meaning is added

    -   It peforms self-attention where tokens can communiate with other tokens to better build context.

    How It Works
    ------------
    
    1, How it works is it takes in a token and looks at it's VECTOR describing it from the

        token embedding table

    2, Tokens now have shape (B,T, num_emb_tk) and each token gets a position VECTOR added to 

        it from the positional embedding table. Token `i` will get VECTOR at `i` in the table

    3, All of the tokens are passed into the self-attention head where they talk to each other
        
        and some tokens find other tokens more interesting so are further more influenced

    4, Finally, all of this is passed into a Linear layer where this is all turned into raw

        logits and returned to CrossEntropyLoss to be evaluated

    Further Information
    -------------------
    -   This is and example of a lookup table,

        `a  |  b  |  c`

        `a 0.5 | 0.9 | 0.7`
        
        `b 0.7 | 0.2 | 0.3`

        `c 0.2 | 0.1 | 0.8`

        If we index `a`, we get a VECTOR of `[[0.5, 0.9, 0.7]]`
    
    
    -   This is information on the returned TENSOR of shape (B,T,C) or similar,

        B is how many batchs we do in parallel, or batch_size

        T is the context size or block_size

        C is the raw VECTOR values for the given token

        
    -   This is information on the prediction and genration training

        The "T" bit works like this

        If we input "hell" and ask to predict the next letter, it doesn't just "o",

        instead it does:

        h -> e

        he -> l

        hel -> l

        hell -> o
        

        So this way it learns order of letters when genrating text.


    Note(s)
    -------
    -   This model genrates letter by letter so all tokens are single characters.
    """

    def __init__(self, vocab_size):
        #basic init
        super().__init__()

        #this creates a token embeding table which is a table with lots of output weights
        #it is of size (vocab_size, num_emb_tk) so that every token has weights to all other tokens
        #when it comes back it has size (1, num_emb_tk) and num_emb_tk is how expressive it is
        self.token_embedding_table = nn.Embedding(vocab_size, num_emb_tk)

        #this creates a embedding table which tells tokens where in the sequence they are
        #this allows the tokens to know where they are and keep order
        #each position has a VECTOR of size (1, num_emb_tk) and tokens at that position get that VECTOR
        self.position_embedding_table = nn.Embedding(block_size, num_emb_tk)

        #make blocks so tokens can go mulitpule rounds of self-attention and feed forward layers
        #a block just applys self-attention, feed forward, residual pathways, layernorm
        #*[] is for list comprehention
        self.blocks = nn.Sequential(*[Block() for _ in range(block_amount)])

        #create layer norm
        self.layer_norm = nn.LayerNorm(num_emb_tk)

        #no use as we have blocks
        # #create self-attention heads
        # #we use mulitpule heads so one head doesn't have to do all of the work
        # self.self_att_heads = MulitHeadAttention(num_heads, num_emb_tk//num_heads)

        # #create a feed foward layer for better token processing
        # self.feed_forward = FeedForwardLayer()

        #create a linear layer to predict the next token
        self.lang_model_head = nn.Linear(num_emb_tk, vocab_size)

    def forward(self, index, answers=None):
        #when we ask the model, it indexs the token and returns a TENSOR
        #the TENSOR shape is (B,T,C)

        #unpack the input TENSOR
        _, T = index.shape

        #if the index has shape (B,T), it returns TENSOR with shape (B,T,C)
        #if the index is a single token, it returns a VECTOR with the token information
        #this returns the token embedding which is information abou the token
        tk_emb = self.token_embedding_table(index)

        #get the position of the tokens
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))

        #add the tk_emb and pos_emb so each token has meaning and position
        tokens = tk_emb + pos_emb

        #pass tokens into blocks
        tokens = self.blocks(tokens)

        #apply layer norm
        tokens = self.layer_norm(tokens)

        #no use as we have blocks
        # #tokens are passed into self-attention to get further meaning
        # tokens = self.self_att_heads(tokens)

        # #send the tokens through a FeedForwardLayer so they can process and understand (next line)
        # #the information gained in attention
        # tokens = self.feed_forward(tokens)

        #plugs the token embeddings into a linear head to predict the next token
        logits = self.lang_model_head(tokens)

        #we check if we have answers to be our target
        #if we don't we aren't training so no need for loss
        if answers == None:
            loss = None
        else:
            #if we have a target the we are training so we need loss
            #reshape the TENSORS into the correct shape
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            answers = answers.view(B*T)

            #now to evaluate the loss function
            loss = F.cross_entropy(logits, answers)

        #return the TENSOR or VECTOR and loss function
        return logits, loss

    def generate(self, index, new_max_tokens):
        #new_max_tokens is the max amount of the tokens the model can generate for one response
        for _ in range(new_max_tokens):
            #only keep the last block_size amount of tokens
            #this is because the model can't see more tokens that block_size
            index_crop = index[:, -block_size:]
            #plug the index, or question, into the model and capture the output
            logits, __ = self(index_crop)
            #crop everything else except the scores for each token for genration
            logits = logits[:, -1, :]
            #apply tempature for a controled spread of different words
            logits = logits / temperature
            #we get the probility for each response using softmax and capture it
            probs = F.softmax(logits, dim=-1)
            #from that we choose the next token and store it
            index_next = torch.multinomial(probs, num_samples=1)
            #then we append it to the current context/index and feed back through the loop
            index = torch.cat((index, index_next), dim=1)

        #when we are finished we return the response
        return index

    def generate_till_target(self, index, target):
        while True:
            index_crop = index[:, -block_size:]
            logits, __ = self(index_crop)
            logits = logits[:, -1, :]
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
            
            if decode(index[0].tolist())[-len(target):] == target:
                break

        return index