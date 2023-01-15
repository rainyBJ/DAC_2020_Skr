def visualize_per_layer(param, title='test', index=0):
    import matplotlib.pyplot as plt
    import os
    # print(os.getcwd())
    channel = 0
    param_list = []

    if(param.shape[channel]>32):
        box_range = 32
    else:
        box_range = param.shape[channel]

    for idx in range(box_range):#param_list记录了每个通道的参数情况 只显示前32个通道的参数
        param_list.append(param[idx].cpu().numpy().reshape(-1))

    fig7, ax7 = plt.subplots()
    ax7.set_title(title)
    ax7.boxplot(param_list, showfliers=False,widths=0.5)
    # plt.ylim(-4, 4)
    # plt.show()
    plt.savefig('weight_distribution/'+str(index)+'_'+title+'.jpg')
    plt.close(fig7)