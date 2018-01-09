
import matplotlib 
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def mnist_4by4_save(samples,path):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05) #이미지 사이간격 조절

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
 
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r',clim=(0.0,1.0))

    plt.colorbar()
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
 
    return None



def gan_loss_graph_save(G_loss,D_loss,path):
    x1 = range(len(G_loss))
    x2 = range(len(D_loss))
    
    y1 = G_loss
    y2 = D_loss

    
    plt.plot(x1,y1,label='G_loss')
    plt.plot(x2,y2,label='D_loss')

    plt.xlabel('weight per update')
    plt.ylabel('loss')
    plt.legend(loc=4)    
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(path)
    return None





 


