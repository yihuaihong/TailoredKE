import numpy as np
import torch

def fine_edit_layer(model, tok, input_tok, lookup_idxs, target_ids):

    source_index = lookup_idxs[0]  #可能有问题，需要测试检查

    hooks = set_act_get_hooks(model, source_index, mlp=True, attn_out=False)

    output = model(input_ids=input_tok[0]['input_ids'], attention_mask=input_tok[0]['attention_mask'], token_type_ids=input_tok[0]['token_type_ids'])
    #output = model(**inp)
    # remove hooks
    remove_hooks(hooks)

    # probs = torch.softmax(output["logits"][:, -1], dim=1)
    # _, attribute_tok = torch.max(probs, dim=1)
    # attribute_tok = attribute_tok.cpu().item()
    # [attribute_tok_str] = decode_tokens(tok, [attribute_tok])

    list_prob = []

    #word_to_encode = 'Apple'  # "Microsoft"  这个是object
    encoded_sequence =  target_ids    #tok.encode(word_to_encode)

    print(f"编码序号：{encoded_sequence}")

    for ix, layer in enumerate(range(len(layers_name))):
        # MLP
        mlp_out = model.activations_[f'm_out_{layer}']  # 在这里每一层都有一个mlp_out，关键就是与microsoft的representation作比较，找到microsoft真正该插入的位置
        # print('mlp_out.shape: ',mlp_out.shape)

        # mlp_out_subject_last = mlp_out[e_range[-1]]
        # print('mlp_out_subject_last.shape: ',mlp_out_subject_last.shape)

        proj = model.lm_head(mlp_out).detach().cpu().numpy()
        # proj = mlp_out.matmul(E.T).cpu().numpy()
        proj_softmax = np.exp(proj) / np.sum(np.exp(proj), axis=0)

        list_prob.append(proj_softmax[encoded_sequence[0]])  #找到对应object的（应该是第一部分的？）的序号

    #然后就从这个list_prob里面通过某个方法找到该被编辑的层就可以了

    return layers_to_edit
        # ind = np.argsort(-proj, axis=-1)
        # attribute_tok_rank = np.where(ind == attribute_tok)[0][0]
        # attribute_tok_score = proj[ind[attribute_tok_rank]]
        # top_k_preds = [decode_tokens(tok, [i])[0] for i in ind[:k]]

        # if ix == 7 or ix == 8 or ix == 9:
        #     print('ix: ', ix, '  top_k_preds: ', top_k_preds)

# project attention and mlp to vocab

# Method
def set_act_get_hooks(model, tok_index, attn=False, attn_out=False, mlp=False, mlp_coef=False):  # mlp
    """
    Works for GPT-J and GPT2 ##Only works on GPT2
    """
    # Make sure that these are not set to True at the same time
    #  so we don't put two different hooks on the same module.
    assert not (attn is True and attn_out is True)

    for attr in ["activations_"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})

    def get_activation(name):
        def hook(module, input, output):
            if "attn" in name:
                if "c_attn" in name:
                    # output.shape: batch_size, seq_len, 3 * hidden_dim
                    _, _, attn_value = output[0].split(model.config.n_embd, dim=1)
                    attn_value = _split_heads(attn_value,
                                              model.config.n_head,
                                              model.config.n_embd // model.config.n_head)
                    model.activations_[name] = attn_value.detach()
                elif "attn_weights" in name:
                    assert len(output) == 3
                    attn_weights = output[2]  # (batch_size, num_heads, from_sequence_length, to_sequence_length)
                    # the last dimension is a distribution obtained from softmax
                    model.activations_[name] = attn_weights[0][:, tok_index, :].detach()
                else:
                    print('output[0].shape: ', output[0].shape)

                    model.activations_[name] = output[0][:, tok_index].detach()
            elif "m_coef" in name:
                # num_tokens = list(input[0].size())[1]  # (batch, sequence, hidden_state)
                model.activations_[name] = input[0][:, tok_index].detach()
            elif "m_out" in name:
                # print('m_out[0].shape: ',output[0].shape)
                model.activations_[name] = output[0][tok_index].detach()

        return hook

    hooks = []
    for i in range(model.config.n_layer):
        if attn is True:
            hooks.append(model.transformer.h[i].attn.c_attn.register_forward_hook(get_activation(f"c_attn_value_{i}")))
            hooks.append(model.transformer.h[i].attn.register_forward_hook(get_activation(f"attn_weights_{i}")))
        if attn_out is True:
            hooks.append(model.transformer.h[i].attn.register_forward_hook(get_activation(f"attn_out_{i}")))
        if mlp_coef is True:
            hooks.append(model.transformer.h[i].mlp.fc_out.register_forward_hook(
                get_activation("m_coef_" + str(i))))  # gpt-j fc_out
        if mlp is True:
            hooks.append(model.transformer.h[i].mlp.register_forward_hook(get_activation("m_out_" + str(i))))

    return hooks


# Always remove your hooks, otherwise things will get messy.
def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()