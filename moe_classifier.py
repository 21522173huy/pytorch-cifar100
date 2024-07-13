
import torch
from torch import nn
import torch.nn.functional as F

class Custom_MLP(nn.Module):
    def __init__(self, in_features, hidden_size = 1024, activation=nn.ReLU(), p = 0.1):
        super(Custom_MLP, self).__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.activation = activation
        
        self.mlp = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            self.activation,
            nn.Dropout(p), 
            nn.Linear(self.hidden_size, self.in_features)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

class MixtralSparseMoeBlock(nn.Module):

    def __init__(self, in_features, hidden_size, p = 0.1):
        super().__init__()
        self.in_features = in_features
        self.hidden_size =  hidden_size
        self.num_experts = 8
        self.top_k = 2

        # gating
        self.gate = nn.Linear(self.in_features, self.num_experts, bias=False)

        self.experts = nn.ModuleList([Custom_MLP(in_features = self.in_features,
                                                       hidden_size = self.hidden_size) for _ in range(self.num_experts)])

        self.dropout = nn.Dropout(p)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        batch_size, hidden_dim = hidden_states.shape

        # router_logits
        router_logits = self.gate(hidden_states) # batch_size, num_experts

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 1:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, hidden_dim)
        final_hidden_states = self.dropout(final_hidden_states) + residual

        return final_hidden_states, router_logits

class MoE_Classifier(nn.Module):
    def __init__(self, in_features, out_features, hidden_size = 1024):
      super(MoE_Classifier, self).__init__()
      self.moe = MixtralSparseMoeBlock(in_features, hidden_size)
      self.classifier = nn.Linear(in_features, out_features)

    def load_balancing_loss_func(self, gate_logits, num_experts = 8, top_k=2, router_aux_loss_coef = 0.001):

      routing_weights = torch.nn.functional.softmax(gate_logits, dim=-1)

      _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

      # treat `top_k` as tokens (shape is `top_k X [batch_size X sequence_length]`)
      selected_experts = selected_experts.reshape(-1)

      expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
      expert_mask = torch.max(expert_mask, dim=-2).values

      # Compute the percentage of tokens routed to each experts
      tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

      # Compute the average probability of routing to these experts
      router_prob_per_expert = torch.mean(routing_weights, dim=0)

      overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(-1))
      return overall_loss * num_experts * router_aux_loss_coef


    def forward(self, features):
      hidden_states, router_logits = self.moe(features)
      prediction = self.classifier(hidden_states)
      aux_loss = self.load_balancing_loss_func(router_logits)
      return (prediction, aux_loss)
