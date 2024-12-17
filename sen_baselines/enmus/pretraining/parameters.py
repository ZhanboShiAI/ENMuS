def get_params(downsample=True):
    params = dict(
        debug = False,
        quick_test = False,     
        finetune_mode = False,  
        pretrained_model_path = "sen_data/models/xxx.h5",

        # Dataset Path
        dataset_dir = "sen_data/target_noiseless",
        metadata_dir = "sen_data/metadata",
        feature_label_dir = 'sen_data/feature_label',
        model_dir = 'sen_data/models/',
        dcase_output_dir = 'sen_data/results/',
        
        mode = 'dev', 
        dataset = 'foa',

        # Feartue params
        fs = 16000,
        hop_len_s = 0.01,

        hop_len = 160,
        win_len = 400,
        nfft = 512,

        label_hop_len_s = 0.1,           
        max_audio_len_s = 60,       
        nb_mel_bins = 64,           
        
        use_salsalite = False,      
        fmin_doa_salsalit = 50,
        fmax_doa_salsalite = 2000,
        fmax_spectra_salsalite = 9000,
        
        # Model type
        multi_accdoa = False,       
        thresh_unify = 15,          
        
        # DNN model prarmeters
        label_sequence_len = 50,        
        batch_size = 64,
        dropout_rate = 0.05,            
        nb_cnn2d_filt = 64,             
        f_pool_size = [4, 2, 2],   
        
        nb_rnn_layers = 2,
        rnn_size = 256,         
        
        self_attn = True,      
        nb_heads = 8,
        
        nb_fnn_layers = 1,
        fnn_size = 128,         
        
        nb_epochs = 500,       
        lr = 1e-3,             
        
        # Metric
        average = 'macro',      
        lad_doa_thresh = 30     
    )

    feature_label_resolution = int(float(str(params['label_hop_len_s'])) // float(str(params['hop_len_s'])))
    params['feature_sequence_len'] = params['label_sequence_len'] * feature_label_resolution

    if downsample:
        params['t_pool_size'] = [int(feature_label_resolution / 5), 1, 1]
    else:
        params['t_pool_size'] = [feature_label_resolution, 1, 1]
    params['patience'] = int(str(params['nb_epochs']))

    params['unique_classes'] = 20

    return params