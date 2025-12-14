# ablation_configs.py
# Mean Flows Table 1 ablations for your current TrainingParams/train.py.
#
# Important:
# - We store embed_t_r_name as a STRING here.
# - run_ablation.py converts embed_t_r_name -> actual lambda and passes embed_t_r.
# - We store time_sampler_params as:
#       None           -> uniform
#       (mean, std)    -> lognorm(mean, std)
# - We store jvp_tangent as a STRING label and map to (bool,bool) in run_ablation.py.


ABLATIONS = {
    "default": {
        "architecture": "DiT-B-4",
        "epochs": 10, 
        "lr": 1e-4,
        "beta1": 0.9,
        "beta2": 0.95,
        "ema_decay": 0.9999,
        "p": 1.0,
        "omega": 1.0,
        "ratio_r_not_eq_t": 0.25,
        "jvp_tangent": "(v,0,1)",
        "embed_t_r_name": "t_tr",
        # IMPORTANT: this key must exist
        # None => uniform, (mean,std) => lognorm(mean,std)
        "time_sampler_params": (-0.4, 1.0),
    },
    "ratio_r_not_eq_t": {"values": [0.00, 0.25, 0.50, 1.00]},
    "jvp_tangent": {"values": ["(v,0,1)", "(v,0,0)", "(v,1,0)", "(v,1,1)"]},
    "embed_t_r_name": {"values": ["tr", "t_tr", "tr_t_tr", "t_tr_only"]},
    # Table 1d sweep uses time_sampler_params
    "time_sampler_params": {
        "values": [
            None,  # uniform
            (-0.2, 1.0),
            (-0.2, 1.2),
            #(-0.4, 1.0),
            (-0.4, 1.2),
        ]
    },
    "p": {"values": [0.0, 0.5, 1.0, 1.5, 2.0]},
    "omega": {"values": [1.0, 1.5, 2.0, 3.0, 5.0]},
    "architecture" : {"values": ["DiT-B-2", "DiT-M-2", "DiT-L-2"]},
    "best": {
        "architecture": "DiT-B-4",
        "epochs": 15, 
        "lr": 1e-4,
        "beta1": 0.9,
        "beta2": 0.95,
        "ema_decay": 0.9999,
        "p": 0.5,
        "omega": 1.5,
        "ratio_r_not_eq_t": 1.00,
        "jvp_tangent": "(v,0,1)",
        "embed_t_r_name": "t_tr",
        # IMPORTANT: this key must exist
        # None => uniform, (mean,std) => lognorm(mean,std)
        "time_sampler_params": (-0.4, 1.2),
    },
}
