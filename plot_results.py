import numpy as np
import matplotlib.pyplot as plt




def main():
    ce_mse_path = 'log/ce_mse_5e-3/'
    ce_iou_path = 'log/ce_iou_5e-3/'
    ce_learned_path = 'log/ce_learned_5e-3/'

    ce_mse_train = np.genfromtxt(ce_mse_path + 'train.csv', delimiter=',')
    ce_mse_val = np.genfromtxt(ce_mse_path + 'val.csv', delimiter=',')

    ce_iou_train = np.genfromtxt(ce_iou_path + 'train.csv', delimiter=',')
    ce_iou_val = np.genfromtxt(ce_iou_path + 'val.csv', delimiter=',')

    ce_learned_train = np.genfromtxt(ce_learned_path + 'train.csv', delimiter=',')
    ce_learned_val = np.genfromtxt(ce_learned_path + 'val.csv', delimiter=',')


    plt.subplots_adjust(hspace=0.6)


    # plt.subplot(3,1,1)
    # plt.title('Train/Val Precision at 5 for all Methods')
    # plt.xlabel('epoch')
    # plt.ylabel('precision at 5')

    # x = range(len(ce_mse_val))
    # y = ce_mse_val[:,1]
    # plt.plot(x,y, 'r', label='ce_mse_val', linestyle='--')
    # x = np.linspace(0, len(ce_mse_val), len(ce_mse_train))
    # y = ce_mse_train[:,1]
    # # plt.plot(x,y, 'r', label='ce_mse_train')

    # x = range(len(ce_iou_val))
    # y = ce_iou_val[:,1]
    # plt.plot(x,y, 'b', label='ce_iou_val', linestyle='--')
    # x = np.linspace(0, len(ce_iou_val), len(ce_iou_train))
    # y = ce_iou_train[:,1]
    # # plt.plot(x,y, 'b', label='ce_iou_train')

    # x = range(len(ce_learned_val))
    # y = ce_learned_val[:,1]
    # plt.plot(x,y, 'orange', label='ce_learned_val', linestyle='--')
    # x = np.linspace(0, len(ce_learned_val), len(ce_learned_train))
    # y = ce_learned_train[:,1]
    # # plt.plot(x,y, 'orange', label='ce_learned_train')

    # plt.legend()
    # plt.show()



    # plt.title('Train/Val IoU for all Methods')
    # plt.xlabel('epoch')
    # plt.ylabel('intersection over union')

    # x = range(len(ce_mse_val))
    # y = ce_mse_val[:,2]
    # plt.plot(x,y, 'r', label='ce_mse_val', linestyle='--')
    # x = np.linspace(0, len(ce_mse_val), len(ce_mse_train))
    # y = ce_mse_train[:,2]
    # #plt.plot(x,y, 'r', label='ce_mse_train')

    # x = range(len(ce_iou_val))
    # y = ce_iou_val[:,2]
    # plt.plot(x,y, 'b', label='ce_iou_val', linestyle='--')
    # x = np.linspace(0, len(ce_iou_val), len(ce_iou_train))
    # y = ce_iou_train[:,2]
    # # plt.plot(x,y, 'b', label='ce_iou_train')

    # x = range(len(ce_learned_val))
    # y = ce_learned_val[:,2]
    # plt.plot(x,y, 'orange', label='ce_learned_val', linestyle='--')
    # x = np.linspace(0, len(ce_learned_val), len(ce_learned_train))
    # y = ce_learned_train[:,2]
    # # plt.plot(x,y, 'orange', label='ce_learned_train')

    # plt.legend()
    # plt.show()



    # plt.subplot(3,1,1)
    plt.title('Train/Val Bounding Box MSE for all Methods')
    plt.xlabel('epoch')
    plt.ylabel('Mean squared error')

    x = range(len(ce_mse_val))
    y = ce_mse_val[:,3]
    plt.plot(x,y, 'r', label='ce_mse_val', linestyle='--')
    x = np.linspace(0, len(ce_mse_val), len(ce_mse_train))
    y = ce_mse_train[:,3]
    # plt.plot(x,y, 'r', label='ce_mse_train')

    x = range(len(ce_iou_val))
    y = ce_iou_val[:,3]
    plt.plot(x,y, 'b', label='ce_iou_val', linestyle='--')
    x = np.linspace(0, len(ce_iou_val), len(ce_iou_train))
    y = ce_iou_train[:,3]
    # plt.plot(x,y, 'b', label='ce_iou_train')

    x = range(len(ce_learned_val))
    y = ce_learned_val[:,3]
    plt.plot(x,y, 'orange', label='ce_learned_val', linestyle='--')
    x = np.linspace(0, len(ce_learned_val), len(ce_learned_train))
    y = ce_learned_train[:,3]
    # plt.plot(x,y, 'orange', label='ce_learned_train')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()