from settings import *

class AtttentionHead(nn.Module):
   """
   Class Information
   -----------------

   This is a single self-attention head, it allows tokens to talk to each other
   which encodes order making the model's prediction more accurate.

   Importiant Information
   ----------------------

   Each token emits 2 vectors; key and query.
   The key vector is what infomation the token contains.
   The query vector is what the token wants or is looking for.
   
   There are 3 Linear layers; key, query, and value:

   Key   - This layer gets all of the keys from the tokens

   Query - This layer gets all of the querys from the tokens
   
   Value - This layer output the model's prediction

   How It Works
   ------------

   1, We first feed the tokens into the key and query Linear layers to get the keys and querys.

   2, Then we take the query and key vectors and matrix multiply them to have tokens talk to each
      other using dot product. 
      
   3, We also have to make sure the results aren't too peaky, or specific values are too large or
      it can interfear with in softmax and other values will be near `0`. 

   4, This is returned into weights and masked so past tokens cant communicate
      with other past tokens that way we create a auto-regressive GPT. 

   5, Each token is asigned a value created by the value layer which will be used next.

   6, Then they are put through a softmax and matrix multiplyed with value (not layer) outputing 
      the final weights which the model will use to more betterly predict the next token.
   """
    
   def __init__(self, head_size):
      #basic init
      super().__init__()

      #create linear layers so we can dot product the keys and querys
      self.key = nn.Linear(num_emb_tk, head_size, bias=False)
      self.query = nn.Linear(num_emb_tk, head_size, bias=False)

      #the value layer chooses the answer
      self.value = nn.Linear(num_emb_tk, head_size, bias=False)

      #create a buffer with the correct masking to tokens cant communicate with the future
      #a buffer is like constant variable but pytorch cares for it
      self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

      #create dropout to prevent overfitting
      self.dropout = nn.Dropout(dropout)

   def forward(self, tokens):
      #unpack the tokens
      B,T,C = tokens.shape

      #get the keys and querys from the tokens into a vector
      key = self.key(tokens)
      query = self.query(tokens)

      #change the shape of the keys and matrix multiply with query to have them talk to each other
      #the *C**-0.5 is to make sure the output isn't too peaky so softmax doesnt choose one anwer too much
      weights = query @ key.transpose(-2, -1) * C**-0.5
      #mask the future so tokens cant see the future
      weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
      #softmax the result
      weights = F.softmax(weights, dim=-1)

      #apply dropout
      weights = self.dropout(weights)

      #get the answers
      value = self.value(tokens)
      #gives each token a weight based on how relevant they are from attention
      output = weights @ value

      #return the output
      return output
   
class MulitHeadAttention(nn.Module):
   """
   Class Information
   -----------------

   This performs mulit head self-attention, the reason for this is so one

   head doesn't have to be able to encode all meaning but instead many heads can

   each encode a different meaning thus deeping meaning and understanding concepts.

   This is also one of the simpler classes.

   Importiant Information
   ----------------------

   -  `nn.ModuleList` is similar to a regular python list, but pytorch cares for it.
       A good example moving devices as well as having it's state saved in state_dict (saving)

   How It Works
   ------------

   1, In the constructer it makes a certain amount of `AttentionHead` classes based on given arguments
       as well as passing head sizes into it

   2, When forward is called, it passes the given tokens into each attention head and gathers the result

   3, Lastly, it concationates the results and returns it. When it concationates the tokens, it adds more
       information so they become more expressive

   """
  
   def __init__(self, num_head, head_size):
      #basic init
      super().__init__()
      #add attention heads to a list for paralell processing
      self.heads = nn.ModuleList([AtttentionHead(head_size) for _ in range(num_head)])
      #projection layer shapes tokens back to the proper size
      self.projection = nn.Linear(num_emb_tk, num_emb_tk)
      #create dropout to prevent overfitting
      self.dropout = nn.Dropout(dropout)

   def forward(self, token):
      #feed the token into many attention heads and concationate the result
      output = torch.cat([head(token) for head in self.heads], dim=-1)
      #projection layer shapes tokens back to the proper size
      output = self.projection(output)
      #apply dropout
      output = self.dropout(output)
      #return the output
      return output
   
class FeedForwardLayer(nn.Module):
   """
   Class Information
   -----------------

   This is a Feed Forward Layer, it is similar to that of a MLP in shape being a

   NeuralNet but smaller. This gives tokens more time to think and deepen understanding 

   as well as adjust to the context better.

   Importiant Information
   ----------------------
   -  This is very similar to a MLP so the mechanics are very similar as well as being simple

   How It Works
   ------------
   1, It creates a Linear-Layer and a relu inside a Sequential.

   2, When forward is called, the tokens get passed through the Linear-Layer and into the relu 
      
      then returned.
   """

   def __init__(self):
      #basic init
      super().__init__()

      #nn.Sequential just pipes the input through it so code is cleaner
      self.neural_net = nn.Sequential(
         #the 4 * is to make the inner layers larger so the tokens gain a deeper understanding and context
         #main Linear-layer that deepens tokens
         nn.Linear(num_emb_tk, 4 * num_emb_tk),
         nn.ReLU(),
         #projection layer shrinks the large output layer back to regular
         nn.Linear(4 * num_emb_tk, num_emb_tk),

         #dropout to prevent overfitting
         nn.Dropout(dropout)
      )

   def forward(self, token):
      #take in a input and put through the neural network then return
      return self.neural_net(token)

class Block(nn.Module):
   """
   Class Information
   -----------------
   This class makes the training process more scaleable as the tokens are passed through

   each block and the amount of blocks can be set. Each tokens goes through these things in 

   this order; self-attention, feed forward layer, layernorm, dropout. There are also residual
   
   connection throughout the block

   Importiant Information
   ----------------------
   -  Layernorms smooth out the influence of each token so no one tokens is too powerfull

   -  Dropout deactivates some precentage of neurons each backward forward pass to ensure the
       network doesn't overfit the training data

   -  The adding of the returned tokens with the orignal tokens is called a residual pathway.
       They are used so the tokens can pile up learning and not have gradents reset each time

   How It Works
   ------------
   1, Each block first creates MultiHeadAttention, FeedForward, and 2 LayerNorm objects for
       later use

   2, Tokens are passed when called and the tokens are passed into the self-attention heads

   3, They get passed into the layernorms to get smoothed out then are returned

   4, Now they are added back to the orignal tokens to form a residual pathway

   5, The same thing is done again but with feedforward and a different layernorm

   6, The tokens are returned back into the model
   """

   def __init__(self):
      super().__init__()
      #make the MulitHeadAttention object
      head_size = num_emb_tk // num_heads
      self.self_att_heads = MulitHeadAttention(num_heads, head_size)
      #make FeedForwardLayer object
      self.feed_forward = FeedForwardLayer()
      #layernorms are used to smooth out the influence / strength of the tokens
      self.layer_norm_1 = nn.LayerNorm(num_emb_tk)
      self.layer_norm_2 = nn.LayerNorm(num_emb_tk)

   def forward(self, tokens):
      #the adding is so the improvments pile up instead of the gradents being reset each time
      #apply layernorms to make sure no token it too powerfull or weak
      #plug the tokens into self-attention heads
      tokens = tokens + self.self_att_heads(self.layer_norm_1(tokens))
      #plug into feed forward so tokens deepen understanding
      tokens = tokens + self.feed_forward(self.layer_norm_2(tokens))

      #return output
      return tokens