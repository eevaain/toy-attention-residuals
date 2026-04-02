import torch
import torch.nn as nn # stateful 
import torch.nn.functional as F # stateless
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader

# tbh maybe i should make this into a jupyter notebook 

class ReverseSequenceDataset(Dataset):
    def __init__(self, num_samples = 5000, T=10, D=16):
        self.X = torch.randn(num_samples, T, D)
        self.Y = torch.flip(self.X, dims=[1]) # flip along the T dimension (sequence)

    def __len__(self): # overriding
        return len(self.X)
    
    def __getitem__(self, idx): # overriding 
        return self.X[idx], self.Y[idx]
    

NUM_SAMPLES = 5000
SEQ_LENGTH = 10
HIDDEN_SIZE = 16
BATCH_SIZE = 32 # 32 samples wihin the 500, so 156 full batches and 1 tiny batch with 8 samples


dataset = ReverseSequenceDataset(num_samples=NUM_SAMPLES, T=SEQ_LENGTH, D=HIDDEN_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True) # builds and streams batches of data from dataset

# input_matrix = dataset.Y
# print(input_matrix)

# length_of_input = dataset.__len__() 
# print(length_of_input)

x_batch, y_batch = next(iter(dataloader)) # forces dataloader to give us the first batch of data. then we decouple the tuple. 

# fun fact: under the hood when you use a for loop, python is really just calling next() over and over again so thats...
# ... why this works. 

# print(f"Input batch shape: {x_batch.shape}") # 32x10x16
# print(f"Target batch shape: {y_batch.shape}") # 32xx10x16

# check if reversal worked

# print("First sequence of tokens in input tensor: ")
# print(x_batch[0, :, 0]) # B = 0, T = all, D = 0 
# print("First sequence of tokens in target tensor: ")
# print(y_batch[0, :, 0])


class SelfAttention(nn.Module): # for a single head of attention. 
    def __init__(self, D, headDim):
        super().__init__()

        self.headDim = headDim

        # split weight matrices by columns (thats why we use headDim), we give each head same X matrix, with weights [D, headDim]
        self.wq = nn.Linear(D, headDim, bias=False) ## nn.linear typically used for when u want to learn new transformations (stateful) ... we wouldnt use einsum here cus einsum is stateless
        self.wk = nn.Linear(D, headDim, bias=False)
        self.wv = nn.Linear(D, headDim, bias=False)

        # TODO: add a KV cache?

    def forward(self, x):
        Q = self.wq(x) # does matmul of [B, T, D] @ [D, headDim] = [B, T, headDim]
        K = self.wk(x) # also [B, T, headDim]
        V = self.wv(x) # also [B, T, headDim]

        logits = torch.einsum("b t d, b s d -> b t s", Q, K) / (self.headDim ** 0.5)
        # s is just a dummy varaible for t (s=t) because einsum needs unique labels for each dimension. 
        #  then do [B, T, D] @ [B, D, T=S] -> [B, T, T=S] (batch of attention score matrices)

        # ^^ since we know we want to preserve batch and NOT mix across batches we can rethink this as:
        # [t, s] @ [s, t] -> [t,t] (we can see the contraction more clearly here, because we sum along the s dimension

        atten_weights = F.softmax(logits, dim=-1) 
        out = torch.einsum("b t s, b s d -> b t d", atten_weights, V)
        return out # [B,T, headDim]
    

class MHA(nn.Module): 
    def __init__(self, D, num_heads=4):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = D // num_heads
        self.D = D

        self.heads = nn.ModuleList([SelfAttention(self.D, self.head_dim) for _ in range(self.num_heads)])
        self.final_proj = nn.Linear(self.D, self.D) # utlimately we want D*D at the end. we keep this in the init cus its stateful! 

    def forward(self, x): 
        head_outputs = [] # [B T D] (after we concat all the head outputs together) 

        for head in self.heads:
            head_out = head(x) # [B,T headDim]
            head_outputs.append(head_out) 

        concatenated_heads = torch.cat(head_outputs, dim=-1) # concat along D of [B, T, headDim] to recover [B,T,D]
        # ^^ fun fact: good practice to do torch.cat outsde of loop because its a costly operation
        return self.final_proj(concatenated_heads) # does [B,T,D] @ [D,D]-> [B,T,D]

# oh i should save my loss runs and gradients before i lose them? make a folder? 

class FullAtnnResLayer(nn.Module): # TODO: plumb in MHA block (DONT PLUMB IT INTO nn.sequential!) 
    def __init__(self, D, num_heads=4): # init is always "build time" 
        super().__init__()


        self.mha_pre_norm = nn.RMSNorm(D) #this will be the norm we use before the mha block 

        self.mha = MHA(D, num_heads)

        self.mlp_pre_norm = nn.RMSNorm(D) # prenorm before mlp block ## RMSNorm is

        self.mha_query = nn.Parameter(torch.zeros(D)) # this 1d vector lives inside a layer permanently. (just a weight vector, called w_l in the paper) FOR ATTNRES NOT REGULAR QUERY   
        ## fun fact ^^ nn.Parameter tells pytorch's memory allocator to allocate physical memory for this tensor PERMANENTLY, and let the autograd engine know that it exists
        ## ^^ if u look at definiton for nn.Linear vs torch.einsum, you'll see that torch.einsum has no nn.parameter! 
        self.norm = nn.RMSNorm(D)

        self.transform = nn.Sequential( # defines one complete MLP block
            nn.Linear(D, D * 2), # does Y = X@W^T + B, we go from D, and upproject to D*2 (weight matrix must be D*2 x D before the transpose). do [B*T,D] @ [D, D*2]
            nn.GELU(),
            nn.Linear(D * 2, D) # down project back to D 
        )

        ## TODO: plumb this into the forward method
        self.mlp_query = nn.Parameter(torch.zeros(D))
    
    def forward(self, previous_states): # previous states is just a list of tensors [h_0, h_1, h_2...] where each h_n is [B,T,D]
        V = torch.stack(previous_states) # [decoder layer (S), B, T, D] (4D tensor)
        # ^^ stack a bunch of h_n together (embedding, layer 1 output, layer 2 output, etc...)
        # also torch.stack is diff than torch.cat. cat glues tensors along an existing dim, stack creates a new dim and then glues. 

        K = self.norm(V) # keys are same as values, but normalized 

        # literally our Q @ K^T 
        logits = torch.einsum("d, s b t d -> s b t", self.mha_query, K) # einsum is just a very flexible way to tensor contractions. here we sum along the hidden dimension D
        # ^^ compare layer's "signature" to a history of previous layers (K)

        attn_weights = F.softmax(logits, dim=0)

        # END OF SCORING PHASE... mixing layer outputs phase next 

        h_l = torch.einsum("s b t, s b t d -> b t d", attn_weights, V) # here we sum along the sequence dim S
        # "give me a vector blend that is 20% of layer 1 and 80% of layer 2..."
        # by summing along the S dimension, we are creating weighted contributions across layers!
        # h_l is the blended output of all the previous layers 

        

        out = self.transform(h_l) # MLP block called here. hm i should put MHA before this line? 


        return out 


class AttentionRoutingModel(nn.Module): # our actual model 
    ## btw we inherit __call__ from nn.Module. python lets u treat objects of that class as if they were functions 
    def __init__(self, D, num_layers=4, use_attn_res=True):
        super().__init__()

        self.use_attn_res = use_attn_res

        self.layers = nn.ModuleList([FullAtnnResLayer(D) for _ in range(num_layers)])

        self.final_proj = nn.Linear(D,D) # does a matmul between [B*T, D] and [D,D] to produce [B*T, D] (our unembedding matrix btw!!)
        # ^^ well technically it just instantiates the weight matrix. the actual matmul happens in self.final_proj's forward method, inherited from nn.Linear. 

    def forward(self, x):
        states = [x] # accumulated embedding + h_n tensors. at first tho we just start with the input embedding, h_0 = x.

        # TODO: add rope or sinusoidal positional encodings? should prolly push what i have first then add pos encodinghs later

        for layer in self.layers:
            if self.use_attn_res:
                out = layer(states) ## attnres on ALSO self.forward calls __call__ method which automatrically triggers the forward method 
                ## ^^ layer is a FullAtnnResLayer object. 
            else:
                out = layer([states[-1]]) ## attnres off (vanilla residual)
        
            states.append(out) # adds h_n to list... -> [h_0 = x, h_1, h_2...]

        return self.final_proj(states[-1]) # final matmul actually happens here! BECAUSE BY PUTTING PARENTHESIS AROUND final_proj, we...
        # ... are invoking the forward method within nn.Linear automatically! (right click on definition of nn.Linear to see what i mean)



### building the training loop 

if __name__ == "__main__":
    USE_ATTN_RES = True ## toggle this to false to see standard res, and true for attnres

    model = AttentionRoutingModel(D=HIDDEN_SIZE, num_layers=4, use_attn_res=USE_ATTN_RES)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    # probe gradients in layer 1 since we have to propagate from last layer back to first. perhaps attnres will help gradients flow back to layer 1 better than standard res? 
    layer1_grads = []
    def capture_gradients(grad):
        layer1_grads.append(grad.abs().mean().item()) # snapshot of a weight gradient i.e. if a weight matrix is (D x D*2), like in our first linear layer, then the gradient is 16x32 = 512 values. 
    
    model.layers[0].transform[0].weight.register_hook(capture_gradients) # this is a hook that triggers capture_gradients every time a gradient flows through the weight matrix of the first linear layer. 

## begin training loop

    epochs = 50
    for epoch in range(epochs): # runs 50 times through the entire dataset (5000 samples)
        epoch_loss = 0.0 # this is just for logging

        for batch_idx, (x_batch, y_batch) in enumerate(dataloader): # runs 157 times. 156 times with 32 samples, and 1 time with 8 samples. this is cus we chose batch_size = 32
            
            optimizer.zero_grad()

            predictions = model(x_batch) # [B*T, D]
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # calcualte avg gradient magnitude for layer 1
        avg_grad = sum(layer1_grads[-len(dataloader):]) / len(dataloader) # grab the last 157 entries within layer1_grads, sum them up and divide by 157! 
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f} | Avg Layer-1 Gradient: {avg_grad:.6f}")

# TODO: implement block attnres 