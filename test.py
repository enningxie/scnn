from networks import *
import os
from data_reader import image_data_set
from test_scnn import *
import theano.tensor as T

if __name__ == '__main__':
    # print("begin")
    test_images_path = "/home/enningxie/Documents/DataSets/true_crowd_counting/val_img"
    test_gt_path = "/home/enningxie/Documents/DataSets/true_crowd_counting/val_gt"
    # networks = [
    #     deep_patch_classifier(),
    #     shallow_net_9x9(),
    #     shallow_net_7x7(),
    #     shallow_net_5x5()
    # ]
    # train_funcs, test_funcs, run_funcs = create_network_functions(networks)
    # print("test_funcs: ", test_funcs)
    # datasets = {
    #     'test': image_data_set(test_images_path, test_gt_path)
    # }
    # print(datasets['test'].do_shuffle)
    # for i, (X, Y) in enumerate(datasets['test']):
    #     print(i, (X, Y))

    # print("---")
    # # _, _, txt = test_scnn(test_funcs, datasets['test'])
    # print("----")
    # # image_files = [f \
    # #                for f in os.listdir(images_path) \
    # #                if os.path.isfile(os.path.join(images_path, f))]
    # # for f in image_files:
    # #     print(os.path.join(gt_path, 'GT_' + os.path.splitext(f)[0] + '.npy'))
    # print('end')

    # test_images_path = './dataset/val_img'
    # test_gt_path = './dataset/val_gt'
    trained_model_files = [
        './models/coupled_train/deep_patch_classifier.pkl',
        './models/coupled_train/shallow_9x9.pkl',
        './models/coupled_train/shallow_7x7.pkl',
        './models/coupled_train/shallow_5x5.pkl'
    ]

    datasets = {
        'test': image_data_set(test_images_path, test_gt_path)
    }
    networks = [
        deep_patch_classifier(),
        shallow_net_9x9(),
        shallow_net_7x7(),
        shallow_net_5x5()
    ]

    load_nets(trained_model_files, networks)
    train_funcs, test_funcs, run_funcs = create_network_functions(networks)

    print('TESTING SCNN...')
    _, _, txt = test_scnn(test_funcs, datasets['test'])
    print(txt)

    print('\n-------\nDONE.')

    # for i, (X, Y) in enumerate(datasets['test']):
    #     print('---')
    #     # print(X, Y)
    #     Y_subnet = test_funcs[1](X)
    #     print('---')



