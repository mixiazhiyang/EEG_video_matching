from util_functions import *
device = 'cuda:6'


def train_step(data_loader, model, opt, bce_loss, epoch, verbose=True):
    mean_acc = 0
    mean_loss = 0
    model.train()
    for p in model.parameters():
        p.requires_grad = True
    _batch_size = 0
    with tqdm(data_loader, disable=not verbose) as pbar:
        for batch_idx, batch in enumerate(pbar):
            eeg = batch['eeg']
            v0 = batch['v0']
            v1 = batch['v1']
            match_label = batch['match_label']
            _batch_size = max(_batch_size, eeg.shape[0])
            eeg = eeg.to(device)
            v0 = v0.to(device)
            v1 = v1.to(device)
            match_label = match_label.to(device)
            # update match model network
            model.zero_grad()
            match_pred = model(eeg, v0, v1)
            match_loss = bce_loss(match_pred, match_label)
            match_loss.backward()
            match_pred_acc = accuracy(match_pred, match_label)
            opt.step()
            # logging
            mean_acc = (match_pred_acc * eeg.shape[0] / _batch_size + mean_acc * batch_idx) / (batch_idx + 1)
            mean_loss = (match_loss.item() * eeg.shape[0] / _batch_size + mean_loss * batch_idx) / (batch_idx + 1)
            if verbose:
                pbar.set_postfix(train='train', epoch=epoch, batch_idx=batch_idx, match_loss=mean_loss,
                                 mean_acc=mean_acc)
    return mean_acc, mean_loss


def eval_step(data_loader, model, bce_loss, epoch=None, verbose=True):
    mean_acc = 0
    mean_loss = 0
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    _batch_size = 0
    with tqdm(data_loader, disable=not verbose) as pbar:
        for batch_idx, batch in enumerate(pbar):
            eeg = batch['eeg']
            v0 = batch['v0']
            v1 = batch['v1']
            match_label = batch['match_label']
            _batch_size = max(_batch_size, eeg.shape[0])
            eeg = eeg.to(device)
            v0 = v0.to(device)
            v1 = v1.to(device)
            match_label = match_label.to(device)
            # update matchmodel network
            match_pred = model(eeg, v0, v1)
            match_loss = bce_loss(match_pred, match_label)
            match_pred_acc = accuracy(match_pred, match_label)
            # logging
            mean_acc = (match_pred_acc * eeg.shape[0] / _batch_size + mean_acc * batch_idx) / (batch_idx + 1)
            mean_loss = (match_loss.item() * eeg.shape[0] / _batch_size + mean_loss * batch_idx) / (batch_idx + 1)
            if verbose:
                pbar.set_postfix(eval='eval', epoch=epoch, batch_idx=batch_idx, match_loss=mean_loss, mean_acc=mean_acc)
    return mean_acc, mean_loss


def test_model(match_model, cp_path, test_loader, bce_loss, ):
    checkpoint = torch.load(cp_path)
    match_model.load_state_dict(checkpoint)
    match_model.eval()
    match_model.to(device)
    test_mean_acc, test_mean_loss = eval_step(test_loader, match_model, bce_loss)
    return test_mean_acc, test_mean_loss


def train_test_model(match_model, train_loader, val_loader, test_loader, exp_name, device=device, if_plot=False,
                     verbose=True, epoch_verbose=True):
    def plot(train_acc_list, eval_acc_list, train_loss_list, eval_loss_list):
        plt.figure()
        plt.plot(train_acc_list)
        plt.plot(eval_acc_list)
        plt.legend(['train', 'dev'])
        plt.title('acc')
        plt.show()
        plt.figure()
        plt.plot(train_loss_list)
        plt.plot(eval_loss_list)
        plt.legend(['train', 'dev'])
        plt.title('loss')
        plt.show()

    cp_path = f'video_match/{exp_name}/model_best.pth'
    json_path = f'video_match/{exp_name}/info.json'
    match_model.to(device)
    optM = optim.Adam(match_model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optM, mode='max', factor=0.1, patience=5, verbose=False)
    early_stopping = EarlyStopping(patience=10, verbose=False)
    num_epochs = 100
    bce_loss = nn.BCELoss()
    train_loss_list = []
    train_acc_list = []
    eval_loss_list = []
    eval_acc_list = []
    learning_rates = []
    disable = not epoch_verbose
    with tqdm(range(num_epochs), disable=disable) as pbar:
        for epoch in pbar:
            train_acc, train_loss = train_step(train_loader, match_model, optM, bce_loss, epoch, verbose=verbose)
            eval_acc, eval_loss = eval_step(val_loader, match_model, bce_loss, epoch, verbose=verbose)
            scheduler.step(eval_acc)
            train_acc_list.append(train_acc)
            eval_acc_list.append(eval_acc)
            train_loss_list.append(train_loss)
            eval_loss_list.append(eval_loss)
            if if_plot:
                plot(train_acc_list, eval_acc_list, train_loss_list, eval_loss_list)
            early_stopping(-eval_acc, match_model, mkdir(cp_path))
            learning_rates.append(optM.param_groups[0]['lr'])
            pbar.set_postfix(epoch=epoch,
                             train_acc=train_acc,
                             train_loss=train_loss,
                             eval_acc=eval_acc,
                             eval_loss=eval_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    test_acc, test_loss = test_model(match_model, cp_path, test_loader, bce_loss)
    save_info = {
        'epoch': epoch + 1,
        'train_loss_list': train_loss_list,
        'train_acc_list': train_acc_list,

        'eval_loss_list': eval_loss_list,
        'eval_acc_list': eval_acc_list,

        'test_loss': test_loss,
        'test_acc': test_acc,

        'learning_rates': learning_rates
    }
    with open(json_path, "w") as f:
        json.dump(save_info, f)
    return save_info


def specific_eval_step(data_loader, model, bce_loss, epoch=None, verbose=True):
    mean_acc = 0
    mean_loss = 0
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    _batch_size = 0
    with tqdm(data_loader, disable=not verbose) as pbar:
        for batch_idx, batch in enumerate(pbar):
            eeg = batch['eeg']
            eeg = torch.ones_like(eeg)
            v0 = batch['v0']
            v1 = batch['v1']
            match_label = batch['match_label']
            _batch_size = max(_batch_size, eeg.shape[0])
            eeg = eeg.to(device)
            v0 = v0.to(device)
            v1 = v1.to(device)
            match_label = match_label.to(device)
            # update matchmodel network
            match_pred = model(eeg, v0, v1)
            match_loss = bce_loss(match_pred, match_label)
            match_pred_acc = accuracy(match_pred, match_label)
            # logging
            mean_acc = (match_pred_acc * eeg.shape[0] / _batch_size + mean_acc * batch_idx) / (batch_idx + 1)
            mean_loss = (match_loss.item() * eeg.shape[0] / _batch_size + mean_loss * batch_idx) / (batch_idx + 1)
            if verbose:
                pbar.set_postfix(eval='eval', epoch=epoch, batch_idx=batch_idx, match_loss=mean_loss, mean_acc=mean_acc)
    return mean_acc, mean_loss


if __name__ == '__main__':
    eeg_signal89 = torch.load('eeg_signal.pth')
    keys_list = np.array(list(eeg_signal89.keys()))

    np.random.shuffle(keys_list)
    all_ids = keys_list
    train_ids = keys_list[:45]  # [53 46 65 26 76 32 19  6 24 49 40 57 34 56 70 15 23 58 78 69 66 55 17  7 1 52  4 73 85 48 36 68 64 10 82 79 86 39 50 25  3 35 28 83 80]
    val_ids = keys_list[45:50] # [41  2 21 47 37]
    test_ids = keys_list[50:]  # [43  8 84 81 14  9]
    print(train_ids, val_ids, test_ids)
    moviefeat_dict = torch.load('movie0.pth')

    save_info_dict = {}

    seg_id = 0
    window_seconds = 3
    sep_seconds = 1
    shift_seconds = 1
    batch_size = 64
    num_workers = 4

    train_data1 = get_data_different_sep_seconds(CustomDataset, moviefeat_dict[seg_id], eeg_signal89, train_ids,
                                                 seg_id=seg_id, window_seconds=window_seconds,
                                                 shift_seconds=shift_seconds, sep_seconds=[1, -7])
    val_data1 = get_data_different_sep_seconds(CustomDataset, moviefeat_dict[seg_id], eeg_signal89, val_ids,
                                               seg_id=seg_id, window_seconds=window_seconds,
                                               shift_seconds=shift_seconds, sep_seconds=[1, -7])
    test_data1 = get_data_different_sep_seconds(CustomDataset, moviefeat_dict[seg_id], eeg_signal89, test_ids,
                                                seg_id=seg_id, window_seconds=window_seconds,
                                                shift_seconds=shift_seconds, sep_seconds=[1, -7])

    train_loader1 = DataLoader(train_data1, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                               collate_fn=col_fn)
    val_loader1 = DataLoader(val_data1, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=col_fn)
    test_loader1 = DataLoader(test_data1, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=col_fn)

    train1_size, dev_size, test_size = split_size(len(moviefeat_dict[seg_id]), [0.8, 0.1, 0.1])
    index_list = np.cumsum([0, train1_size, dev_size, test_size])
    train_data2 = get_data_different_time(CustomDataset, moviefeat_dict[seg_id], eeg_signal89, all_ids, seg_id=seg_id,
                                          se_img_index=[[index_list[0], index_list[1]]], window_seconds=window_seconds,
                                          shift_seconds=shift_seconds, sep_seconds=[1, -7])
    val_data2 = get_data_different_time(CustomDataset, moviefeat_dict[seg_id], eeg_signal89, all_ids, seg_id=seg_id,
                                        se_img_index=[[index_list[1], index_list[2]]], window_seconds=window_seconds,
                                        shift_seconds=shift_seconds, sep_seconds=[1, -7])
    test_data2 = get_data_different_time(CustomDataset, moviefeat_dict[seg_id], eeg_signal89, all_ids, seg_id=seg_id,
                                         se_img_index=[[index_list[2], index_list[3]]], window_seconds=window_seconds,
                                         shift_seconds=shift_seconds, sep_seconds=[1, -7])

    train_loader2 = DataLoader(train_data2, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                               collate_fn=col_fn)
    val_loader2 = DataLoader(val_data2, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=col_fn)
    test_loader2 = DataLoader(test_data2, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=col_fn)

    model_names = ['DilationModel', 'DilationVideoGRUModel', 'DilationVideoLSTMModel', 'DilationTransformerModel',
                   'CNNTransformerModel', 'CNNTimeEmbTransformerModel', 'DilationGRUModel', 'DilationGRUEEGModel',
                   'CNNGRUTransformerModel', 'CNNTransformer1HeadVideoGRULowDimModel',
                   'CNNTransformer1HeadVideoGRUModel', 'OneWayDilationVideoGRUModel']

    for i, name in enumerate(model_names):
        match_model = init_models(name)
        num = count_trainable_param(match_model)
        print(i + 1, name, np.round(num, 2))

    # reproduce experiments 12 models table
    for i, name in enumerate(model_names):
        exp_name = f'{name}_exp1'
        match_model = init_models(name)
        save_info_dict[exp_name] = train_test_model(match_model, train_loader=train_loader1, val_loader=val_loader1,
                                                    test_loader=test_loader1, exp_name=exp_name, verbose=False)

    # most fair dataset, every match has two mismatch in two directions, not shown in paper
    seg_id = 0
    window_seconds = 3
    shift_seconds = 1
    sep_seconds = 1
    train_data3, val_data3, test_data3 = [TwoSideDataset(moviefeat_dict[seg_id],
                                                         eeg_signal89,
                                                         _id_list,
                                                         seg_id=seg_id,
                                                         window_seconds=window_seconds,
                                                         shift_seconds=shift_seconds,
                                                         sep_seconds=sep_seconds) for _id_list in
                                          [train_ids, val_ids, test_ids]]
    train_loader3, val_loader3, test_loader3 = [
        DataLoader(_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=col_fn) for _data in
        [train_data3, val_data3, test_data3]]
    exp_name = 'DilationVideoGRUModel_exp3'
    match_model = init_models('DilationVideoGRUModel')
    save_info_dict[exp_name] = train_test_model(match_model, train_loader=train_loader3, val_loader=val_loader3,
                                                test_loader=test_loader3, exp_name=exp_name, verbose=False)

    # balance and imbalance dataset experiment
    # imbalance dataset, only one mismatch after match

    train_data4 = get_data_different_sep_seconds(CustomDataset, moviefeat_dict[seg_id], eeg_signal89, train_ids,
                                                 seg_id=seg_id, window_seconds=window_seconds,
                                                 shift_seconds=shift_seconds, sep_seconds=[1, ])
    val_data4 = get_data_different_sep_seconds(CustomDataset, moviefeat_dict[seg_id], eeg_signal89, val_ids,
                                               seg_id=seg_id, window_seconds=window_seconds,
                                               shift_seconds=shift_seconds, sep_seconds=[1, ])
    test_data4 = get_data_different_sep_seconds(CustomDataset, moviefeat_dict[seg_id], eeg_signal89, test_ids,
                                                seg_id=seg_id, window_seconds=window_seconds,
                                                shift_seconds=shift_seconds, sep_seconds=[1, ])

    train_loader4 = DataLoader(train_data4, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                               collate_fn=col_fn)
    val_loader4 = DataLoader(val_data4, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=col_fn)
    test_loader4 = DataLoader(test_data4, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=col_fn)

    exp_name = 'DilationVideoGRUModel_exp4'
    match_model = init_models('DilationVideoGRUModel')
    save_info_dict[exp_name] = train_test_model(match_model, train_loader=train_loader4, val_loader=val_loader4,
                                                test_loader=test_loader4, exp_name=exp_name, verbose=False)

    match_model1 = load_exp_model('DilationVideoGRUModel_exp1').to('cuda:6')
    match_model4 = load_exp_model('DilationVideoGRUModel_exp4').to('cuda:6')
    test_sep_seconds_list = np.arange(-23, 20)
    acc_list1 = []
    loss_list1 = []
    acc_list4 = []
    loss_list4 = []
    seg_id = 0
    for test_sep_seconds in test_sep_seconds_list:
        evaluation_loader1 = DataLoader(
            CustomDataset(moviefeat_dict[seg_id],
                          eeg_signal89,
                          test_ids, seg_id=seg_id,
                          window_seconds=3, sep_seconds=test_sep_seconds, ), batch_size=batch_size, shuffle=True,
            num_workers=0, collate_fn=col_fn)
        evaluation_acc1, evaluation_loss1 = eval_step(evaluation_loader1, match_model1, nn.BCELoss())
        evaluation_acc4, evaluation_loss4 = eval_step(evaluation_loader1, match_model4, nn.BCELoss())
        acc_list1.append(evaluation_acc1)
        loss_list1.append(evaluation_loss1)
        acc_list4.append(evaluation_acc4)
        loss_list4.append(evaluation_loss4)

    plt.figure(figsize=(5, 5), dpi=300)
    plt.plot(test_sep_seconds_list, acc_list1)
    plt.plot(test_sep_seconds_list, acc_list4)
    plt.legend(['balance', 'imbalance'])
    plt.xlabel('sep seconds/s')
    plt.ylabel('accuracy/%')
    plt.vlines(-3, 0.4, 0.7, 'g')
    plt.hlines(0.5, min(test_sep_seconds_list), max(test_sep_seconds_list), 'r')
    plt.text(-20, 0.51, 'y=0.5')
    plt.text(-2, 0.45, 'x=-3')
    plt.savefig('paper/wrong_accuracy_and_sep_seconds.png')
    plt.savefig('paper/wrong_accuracy_and_sep_seconds.eps')
    plt.show()

    # experiments for some fun
    match_model = load_exp_model('OneWayDilationVideoGRUModel_exp1').to('cuda:6')

    seg_id = 0

    train_eeg_data1 = CustomDataset(moviefeat_dict[seg_id],
                                    eeg_signal89,
                                    train_ids,
                                    seg_id=seg_id,
                                    window_seconds=3,
                                    shift_seconds=3,
                                    sep_seconds=-3)
    train_eeg_loader1 = DataLoader(train_eeg_data1, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_eeg_feature_list, train_video_feature_list, train_img_idx_list = get_eeg_features(train_eeg_loader1,
                                                                                            match_model, device)
    print(train_eeg_feature_list.shape, train_video_feature_list.shape, train_img_idx_list.shape)
    test_eeg_data1 = CustomDataset(moviefeat_dict[seg_id],
                                   eeg_signal89,
                                   test_ids,
                                   seg_id=seg_id,
                                   window_seconds=3,
                                   shift_seconds=3,
                                   sep_seconds=-3)
    test_eeg_loader1 = DataLoader(test_eeg_data1, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_eeg_feature_list, test_video_feature_list, test_img_idx_list = get_eeg_features(test_eeg_loader1, match_model,
                                                                                         device)

    # calculate importance of each channel
    match_model = load_exp_model('OneWayDilationVideoGRUModel_exp1').to('cuda:6')
    mean_importance = 0
    for i in tqdm(range(len(test_eeg_data1))):
        importance = channel_gradients(match_model.to('cpu'), test_eeg_data1[i])
        importance = importance.numpy()
        mean_importance = (mean_importance * i + importance) / (i + 1)

    with open('EEG_data_info.pkl', 'rb') as f:
        info = pickle.load(f)

    keys = list(info.get_montage().get_positions()['ch_pos'].keys())
    pos = np.stack(info.get_montage().get_positions()['ch_pos'].values(), 0)
    pos[:, :2] = -pos[:, :2]

    sensorPosition = dict(zip(keys, pos))  # 制定为字典的形式
    myMontage = mne.channels.make_dig_montage(ch_pos=sensorPosition)

    info = mne.create_info(
        ch_names=keys,
        ch_types=['eeg'] * 64,  # 通道个数
        sfreq=1000)  # 采样频率
    info.set_montage(myMontage)

    scaled_importance = (np.array(mean_importance) - np.min(mean_importance)) / (
            np.max(mean_importance) - np.min(mean_importance)) * 10
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    im, cn = mne.viz.plot_topomap(scaled_importance,
                                  info, axes=ax,
                                  names=keys,
                                  # vlim=(-2, 2)
                                  show=False, cmap='jet'
                                  )
    plt.colorbar(im)
    plt.savefig('paper/avg_heat_map_over_eeg_channels.eps')
    plt.savefig('paper/avg_heat_map_over_eeg_channels.png')
    plt.show()

    match_model.to('cpu')
    test_feature_cos = F.cosine_similarity(test_eeg_feature_list, test_video_feature_list, dim=-1)
    test_matchable = F.sigmoid(match_model.fc_layer(test_feature_cos))
    train_feature_cos = F.cosine_similarity(train_eeg_feature_list, train_video_feature_list, dim=-1)
    train_matchable = F.sigmoid(match_model.fc_layer(train_feature_cos))
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(np.arange(71), train_matchable.reshape([-1, 71]).permute(1, 0).mean(-1))
    plt.plot(np.arange(71), test_matchable.reshape([-1, 71]).permute(1, 0).mean(-1))
    plt.hlines(0.5, 0, 71, color='r', linestyles='--')
    plt.xlim([0, 71])
    seg_list = np.arange(0, 71, 5)
    sec_list = [str(i) for i in np.arange(0, 71, 5) * 3]
    plt.xticks(seg_list, sec_list)
    plt.legend(['train', 'test'])
    plt.xlabel('t/s')
    plt.ylabel('prediction')
    train_acc = np.round((sum(train_matchable > 0.5) / len(train_matchable)).item() * 100, 2)
    test_acc = np.round((sum(test_matchable > 0.5) / len(test_matchable)).item() * 100, 2)
    plt.text(5, 0.85, f'train accuracy:{train_acc}%\ntest accuracy:{test_acc}%', fontsize=12, color='black')
    plt.savefig('paper/train_test_feature_OneWayDilationVideoGRUModel_exp1_eeg_video_match_prediction_mean.eps')
    plt.show()
    print(sum(train_matchable > 0.5) / len(train_matchable))
    print(sum(test_matchable > 0.5) / len(test_matchable))

    # video dino TSNE plot

    scaler = StandardScaler()
    X = moviefeat_dict[0]
    X = scaler.fit_transform(X)
    pca = TSNE(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10, 5), dpi=300)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=np.arange(X.shape[0]), cmap='jet', s=0.5)
    plt.colorbar()
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(mkdir('paper/dino_feature.png'))
    plt.savefig(mkdir('paper/dino_feature.eps'))
    plt.show()

    # video reconstruction
    # EEG video match reconstruction
    EEG_video_matching_accuracy = evaluate_matching_accuracy(test_eeg_feature_list,
                                                             test_video_feature_list[:test_eeg_data1.seg_num],
                                                             model=match_model,
                                                             k_list=[1, 5, 10])
    print('EEG video match reconstruction accuracy:', EEG_video_matching_accuracy)
    # EEG order reconstruction
    func = lambda x: x.reshape(x.shape[0], -1)
    pred, frame_errors, quantized_acc = decoding_video(train_eeg_feature_list, train_img_idx_list,
                                                       test_eeg_feature_list, test_img_idx_list, func, allow_arr=25 * 1)
    print('EEG order reconstruction accuracy:', quantized_acc)

    random_guessing_video_recover_accuracy = random_guessing(test_img_idx_list.numpy(), allow_arr=25 * 1)
    print('EEG order random reconstruction accuracy:', random_guessing_video_recover_accuracy)

    # silhouette score table
    ss_table = np.zeros([4, 2])
    extractor = FeatureExtractor()
    feat_all = []
    person_all = []
    for idx, item in enumerate(test_eeg_data1):
        eeg = item['eeg']
        pid = item['person_nnid']
        eeg = eeg / torch.max(torch.abs(eeg)) * 0.8
        eeg_feat = extractor.extract(eeg.detach().cpu().numpy())
        feat_all.append(eeg_feat)
        person_all.append(pid)
    feat_all_tensor = np.stack(feat_all, axis=0)

    # segment level person traditional features
    test_person_labels = np.repeat(np.arange(6), 71)
    test_eeg_feature_cluster = feat_all_tensor.copy()
    ss_table[0, 0] = silhouette_score(test_eeg_feature_cluster, test_person_labels, metric='cosine')

    # segment level position traditional features
    test_seg_labels = np.tile(np.arange(71), 6)
    test_eeg_feature_cluster = feat_all_tensor.copy()
    ss_table[0, 1] = silhouette_score(test_eeg_feature_cluster, test_seg_labels, metric='cosine')

    # segment level person deep features
    test_person_labels = np.repeat(np.arange(6), 71)
    test_eeg_feature_cluster = test_eeg_feature_list.reshape([-1, 256 * 75])
    ss_table[1, 0] = silhouette_score(test_eeg_feature_cluster, test_person_labels, metric='cosine')

    # segment level position deep features
    test_seg_labels = np.tile(np.arange(71), 6)
    test_eeg_feature_cluster = test_eeg_feature_list.reshape([-1, 256 * 75])
    ss_table[1, 1] = silhouette_score(test_eeg_feature_cluster, test_seg_labels, metric='cosine')

    # frame level postion deep features
    test_postion_labels = np.tile(np.repeat(np.arange(71), 75), 6)
    test_eeg_feature_cluster = torch.permute(test_eeg_feature_list, (0, 2, 1)).contiguous().reshape([-1, 256])
    ss_table[2, 0] = silhouette_score(test_eeg_feature_cluster, test_postion_labels, metric='cosine')

    # frame level person deep features
    test_person_labels = np.repeat(np.arange(6), 71 * 75)
    test_eeg_feature_cluster = torch.permute(test_eeg_feature_list, (0, 2, 1)).contiguous().reshape([-1, 256])
    ss_table[2, 1] = silhouette_score(test_eeg_feature_cluster, test_person_labels, metric='cosine')

    # frame level position DINO features
    unit = 75
    max_label_num = int(moviefeat_dict[0].shape[0] / unit)
    test_eeg_feature_cluster = moviefeat_dict[0][:unit * max_label_num]
    test_seg_labels = np.tile(np.arange(unit), max_label_num)
    ss_table[3, 1] = silhouette_score(test_eeg_feature_cluster, test_seg_labels, metric='cosine')
    print('silhouette score table')
    print(ss_table)
    # silhouette plots
    # traditional
    test_person_labels = np.repeat(np.arange(6), 71)
    test_eeg_feature_cluster = feat_all_tensor.copy()
    plot_silhouette_score(test_eeg_feature_cluster, test_person_labels, 'paper/traditional_person_feature_cluster.eps')

    # ours
    test_person_labels = np.repeat(np.arange(6), 71)
    test_eeg_feature_cluster = test_eeg_feature_list.reshape([-1, 256 * 75])

    plot_silhouette_score(test_eeg_feature_cluster, test_person_labels, 'paper/our_person_feature_cluster.eps')
