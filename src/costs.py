import tensorflow as tf

def get_costs(a, x_gen, x_feed, d_gen, d_feed, sig_d_feed, sig_d_gen, d_interp, x_interp):
    #Standard epsilon for zero-division overflow
    R_eps = 1e-18
    #Standard MM-nsat R rescaling factor
    mean_sig_d_gen = tf.reduce_mean(sig_d_gen)
    R_mm_nsat = (1.0 + R_eps - mean_sig_d_gen) / (mean_sig_d_gen + R_eps)

    ##STANDARD IMPLEMENTATION OF CROSS ENTROPIES##
    msce_real_1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_feed), logits=d_feed)
    msce_real_0 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_feed), logits=d_feed)
    msce_gen_0 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_gen), logits=d_gen)
    msce_gen_1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_gen), logits=d_gen)
    
    #Standard logistic discriminator cost: NS-GAN, MM-GAN, MM-nsat, ...
    def d_cost_ns(): return msce_real_1 + msce_gen_0
    #Softplus implementation of logistic discriminator, not used
    def d_cost_ns_softplus(): return tf.nn.softplus(d_gen) + tf.nn.softplus(-d_feed)
    #Logistic discriminator with zero-centred gradient penalty: TODO efficient gradient computation?
    def d_cost_ns_simplegp():
        #R1-gamma controlled by a.d_cost_parameter
        loss = msce_real_1 + msce_gen_0
        real_loss = tf.reduce_sum(d_feed)
        real_grads = tf.gradients(real_loss, [x_feed])[0]
        r1_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
        loss += r1_penalty * (a.d_cost_parameter * 0.5)
        return loss

    #Other standard discriminators
    def d_cost_hinge(): return tf.nn.relu(1.0 - d_feed) + tf.nn.relu(1.0 + d_gen)
    def d_cost_wgan(): return -d_feed + d_gen
    def d_cost_lin_ls(): return 0.5 * (tf.square(d_feed-1.0) + tf.square(d_gen-0.0))
    def d_cost_sig_ls(): return 0.5 * (tf.square(sig_d_feed-1.0) + tf.square(sig_d_gen-0.0))
    def d_cost_wgan_gp(gp_target=1.0, gp_weight=a.d_cost_parameter):
        cost_adv = -d_feed + d_gen
        interp_grads = tf.gradients(tf.reduce_sum(d_interp), [x_interp])[0]
        interp_norms = tf.sqrt(tf.reduce_sum(tf.square(interp_grads), axis=[1,2,3]))
        cost_penalty = tf.square(interp_norms - gp_target) * gp_weight
        return cost_adv + cost_penalty

    #Mapping dictionary: argparse string -> cost function
    d_cost_dict = {'ns':d_cost_ns,'mm':d_cost_ns,'hinge':d_cost_hinge,'wgan_gp':d_cost_wgan_gp,'lin_ls':d_cost_lin_ls,'sig_ls':d_cost_sig_ls,'ns_sgp':d_cost_ns_simplegp,'ns_softplus':d_cost_ns_softplus}
    disc_loss = d_cost_dict[a.d_cost]()

    #Standard MM-GAN and NS-GAN generator costs
    def g_cost_mm(): return -msce_gen_0
    def g_cost_ns(): return msce_gen_1
    def g_cost_mm_softplus(): return -tf.nn.softplus(d_gen)
    def g_cost_ns_softplus(): return tf.nn.softplus(-d_gen)
    #Other standard generator costs
    def g_cost_wgan(): return -d_gen
    def g_cost_hinge(): return -d_gen
    def g_cost_lin_ls(): return 0.5 * tf.square(d_gen-1.0)
    def g_cost_sig_ls(): return 0.5 * tf.square(sig_d_gen-1.0)

    # (0+a) * NS-GAN + (1-a)*MM-nsat, where a==a.g_cost_parameter from argparse
    def g_cost_nsmmnsat():
        return a.g_cost_parameter * g_cost_mm() *tf.stop_gradient(R_mm_nsat) + (1.0 - a.g_cost_parameter) * g_cost_ns()
    # (0+a) * NS-GAN + (1-a)*MM-GAN, where a==a.g_cost_parameter from argparse
    def g_cost_nsmm_imba():
        return a.g_cost_parameter * g_cost_mm() + (1.0 - a.g_cost_parameter) * g_cost_ns()
    # Experimental constant influence cost
    def g_cost_cinf():
        R = (1.0 + R_eps - mean_sig_d_gen)**2 / (mean_sig_d_gen + R_eps)
        return tf.stop_gradient(R) * -sig_d_gen / (1.0 + a.g_cost_parameter - sig_d_gen)

    #Exotic generator costs: used for scaling factor experiments, less well tested than other costs
    def g_cost_nssq():
        cost_ns = msce_gen_1
        cost_add = -sig_d_gen
        R = tf.reduce_mean(1. / (1. - sig_d_gen + R_eps))
        R = tf.clip_by_value(R, 1e-8, 1e8)
        return (cost_ns + cost_add) * tf.stop_gradient(R)
    def g_cost_mmsq(): 
        cost_mm = -msce_gen_0
        cost_add = sig_d_gen
        dp_bar = tf.reduce_mean(sig_d_gen)
        R = (1. + R_eps - dp_bar) / (R_eps + dp_bar)**2
        return (cost_mm + cost_add) * tf.stop_gradient(R)
    def g_cost_nsadd():
        cost_ns = msce_gen_1
        cost_add = -d_gen * a.g_cost_parameter
        R = (1 + R_eps - mean_sig_d_gen) / (1 + R_eps + a.g_cost_parameter - mean_sig_d_gen)
        return (cost_ns + cost_add) * tf.stop_gradient(R)
    def g_cost_mmadd():
        cost_mm = -msce_gen_0
        cost_add = -d_gen * a.g_cost_parameter
        R = (1 + R_eps - mean_sig_d_gen) / (R_eps + a.g_cost_parameter + mean_sig_d_gen)
        return (cost_mm + cost_add) * tf.stop_gradient(R)
    def g_cost_nssqrt():
        x = tf.clip_by_value(d_gen, -20, 20)
        prelog = tf.sqrt(tf.exp(-x)) + tf.sqrt(tf.exp(-x)+1)
        cost = 2*tf.log(prelog)
        R = tf.math.sqrt(1 + R_eps - mean_sig_d_gen)
        return cost * tf.stop_gradient(R)
    def g_cost_mmsqrt():
        x = tf.clip_by_value(d_gen, -20, 20)
        prelog = tf.sqrt(tf.exp(x)) + tf.sqrt(tf.exp(x)+1)
        cost = -2*tf.log(prelog)
        R = (1 + R_eps - mean_sig_d_gen) / tf.math.sqrt((R_eps + mean_sig_d_gen))
        return cost * tf.stop_gradient(R)

    g_cost_dict = {'mm':g_cost_mm,'ns':g_cost_ns,'mm_softplus':g_cost_mm_softplus,'ns_softplus':g_cost_ns_softplus,
        'ns_mmnsat':g_cost_nsmmnsat,'nsmm_imba':g_cost_nsmm_imba,
        'lin_ls':g_cost_lin_ls,'sig_ls':g_cost_sig_ls,'wgan':g_cost_wgan,'hinge':g_cost_hinge,
        'mmsq':g_cost_mmsq,'nssq':g_cost_nssq,'nsadd':g_cost_nsadd,'mmadd':g_cost_mmadd,'mmsqrt':g_cost_mmsqrt,'nssqrt':g_cost_nssqrt,'cinf':g_cost_cinf}
    gen_loss = g_cost_dict[a.g_cost]()

    return disc_loss, gen_loss
