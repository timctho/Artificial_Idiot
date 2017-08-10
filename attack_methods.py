from cleverhans import attacks
import cleverhans


def fgsm(model, max_epsilon, clip_min, clip_max):

    attacker = attacks.FastGradientMethod(model)
    x_adv = attacker.generate(model.prepro_input_tensor,
                              eps=max_epsilon,
                              clip_min=clip_min,
                              clip_max=clip_max)
    return x_adv


def basic_iterative(model, eps, clip_min, clip_max):
    attacker = attacks.BasicIterativeMethod(model)
    x_adv = attacker.generate(model.prepro_input_tensor,
                                  eps=eps,
                                  clip_min=clip_min,
                                  clip_max=clip_max)
    return x_adv