from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import repr_tools
from util import nethook

from .memit_hparams import MEMITHyperParams
from .dynamic_layers import fine_edit_layer


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")[  #所谓的target就是object
        "input_ids"
    ][0]


    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context_types in context_templates
        for context in context_types
    ], ["{} is a"]

    if 'rephrase_sentences_' in request: #当request里有这一项时
        rephrase_prompts = [rephrase_sentence + tok.decode(target_ids[:-1]) for rephrase_sentence in request['rephrase_sentences_']]
        rewriting_prompts += rephrase_prompts  # 可能直接在这里添加即可？？


    all_prompts = rewriting_prompts + kl_prompts

    print('all_prompts: ',all_prompts)

    input_tok = tok(      #input_tok的length比原来多了一个的原因是因为kl_prompts的原因
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    print('input_tok: ',input_tok) #需要知道input_tok里面到底有哪些key

    print('input_tok["input_ids"].shape: ',input_tok["input_ids"].shape)
    print('input_tok["attention_mask"].shape: ', input_tok["attention_mask"].shape)

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(  #这里是预测的object的位置，是用来后面计算prediction的loss的
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids  #而且这里也是默认说object的位置一定是在句子末尾的

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [   #lookup_idxs里面也包括了kl_prompts的last_subject的位置
        find_fact_lookup_idx(    #已经检查过，这个函数是通过中括号{}来查找subject_token对应位置，因此prompt的部分包括subject也没关系，只要没有{}
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]


    #在这里加个函数，通过可解释性、梯度、或相似性的方法去找该被编辑的层
    layers_to_edit = fine_edit_layer(
            model,
            tok,
            input_tok,
            lookup_idxs,
            target_ids,
    )



    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)

    """
    print(f"Rewrite layer is {layer}")
    """
    print(f"After location, Rewrite layer is {layers_to_edit}")  #暂时假如只有一层
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    delta = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda")  #关于delta的初始化，考虑换成其他方式会不会有更好的效果？ delta本来为不修改的  我们可以考虑在这基础上赋予不同的attribute？

    delta_list = [torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda") for _ in range(27)]

    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        global prompt_target #声明这个根据prompt得来的hiddenstate为全局变量，方便取出使用

        print('cur_layer: ',cur_layer)  #为什么这里的循环不是从1开始到最后一层？
        print('layer: ', layer)

        #if cur_layer == hparams.layer_module_tmp.format(layer): #这里的layer是那个17层的layer，中间层的layer
        if cur_layer == hparams.layer_module_tmp.format(layers_to_edit):  # 这里的layer是那个17层的layer，中间层的layer
            # Store initial value of the vector of interest


            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()  #lookup_idxs[0]是第一个句子，最开始没有context_templates的句子

                """

                print('cur_out[0].shape: ',cur_out[0].shape)

                print('lookup_idxs[-2]: ',lookup_idxs[-2])
                # 这里为本来设计的prompt给后的subject token的hiddenstate
                #目前关键就在这里 只在target_init is None时记录一次就好，但在其他step记录的结果也应该是一样的，同时因为optimize的只有delta，所以计算图应该是很小的？ 如何查看衡量计算图的大小？
                prompt_target = cur_out[0][-2, lookup_idxs[-2], :].detach().clone()  #暂时应该是这样 意思就是中间层应该是这样的hiddenstate？？才合适，才能映射出对应那个新知识

                #然后找到ike_32的结果去替换一下 暂时替换了这句“Imagine that LeBron James is a xx now”
                """

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):  #只在lookup_ids的位置添加delta，也就是只在last_subject的位置添加delta
                cur_out[0][i, idx, :] += delta   #这里是在每句话last_subject的位置添加delta   #0 #暂时去掉delta，评估在哪一层的gradient比较大 #delta  #每一个prompt的hiddenstate都被相同的delta给修改了

        return cur_out

    """
    mlp_params = []
    for name, param in model.named_parameters():
        if 'mlp.c_proj' in name:
            mlp_params.append(param)

    print('mlp_params: ',mlp_params)
    """


    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)  #优化的参数只有一个，就是delta，只有delta

    # opt = torch.optim.Adam(mlp_params, lr=hparams.v_lr)  #优化的参数只有一个，就是delta，只有delta

    nethook.set_requires_grad(False, model)

    # # 设置mlp_params中参数的requires_grad为True
    # for param in mlp_params:
    #     param.requires_grad = True

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
                hparams.layer_module_tmp.format(1),
                hparams.layer_module_tmp.format(2),
                hparams.layer_module_tmp.format(3),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            print('kl_logits.shape: ',kl_logits.shape)
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        full_repr = tr[hparams.layer_module_tmp.format(loss_layer)].output[0][  #这里是在loss_layer层，也即是在最后一层输出后的loss
            : len(rewriting_prompts)
        ]

        print('full_repr.shape: ',full_repr.shape)

        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w + lm_b, dim=2)  #这里应该就是某个中间的hiddenstate，然后点乘矩阵的权重，并且加上bias，得到一个输出的概率
        print('log_probs.shape: ', log_probs.shape)


        #在训练阶段是已经有每个位置应该有的输出，因此可以把每个位置的该有的token同时直接输入，利用只往前看的三角attention矩阵，最后同时计算一次loss值即可


        loss = torch.gather(  #这里是match的，没错的，确实自回归supervised learning的时候不用输入最后一个token的hiddenstate，只需要输入最后一个之前的，然后这里的target_ids是准的，是对的
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2), #明白了，所以这里是类似一个supervised learning
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay

        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']['str']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )


        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()

        print('delta.grad: ',delta.grad)
        # 梯度查看
        # for name, param in model.named_parameters():
        #     if 'mlp.c_proj.weight' in name:
        #         print(f"Parameter: {name}, l2norm_Gradient: {torch.norm(param.grad)}, l1norm_Gradient: {torch.norm(param.grad,p=1)}")


        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta   #所以修改的地方就可以在这里，原来的方案为：target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()
                                   #然后往这个上面加上delta
                                   #那假如我能直接找到一个更加合适的target呢，也即是cur_out应该有的值

    print('target.shape: ',target.shape)
    print('target: ',target)
    print('target_init: ',target_init)



    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    """
    print('cos_similarity between target_init and target: ',torch.nn.functional.cosine_similarity(target_init, target, dim=0))
    print('cos_similarity between target_init and prompt_target: ',torch.nn.functional.cosine_similarity(target_init, prompt_target, dim=0))

    print('Euclidean similarity between target_init and target: ', torch.norm(target_init - target, p=2))
    print('Euclidean similarity between target_init and prompt_target: ', torch.norm(target_init - prompt_target, p=2))

    # random_target = torch.rand(1600).to("cuda")
    """

    return target
    #return target * 0.4 + prompt_target * 0.6 #(target+prompt_target)/2
    #return target


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
