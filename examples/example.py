import matplotlib.pyplot as plt
import tensorflow as tf

from smelu import smelu


def main():
    x = tf.linspace(-6, 6, 1000)
    
    fig, axs = plt.subplots(1, 2)
    for beta in [0.1, 0.5, 1, 2, 10]:
        # Gradients
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = smelu(x, beta=beta)  # dy = SmeLU(x) dx
            dy_dx = tape.gradient(y, x)
        
        axs[0].plot(x, y, label=f"β={beta}")
        axs[1].plot(x, dy_dx, label=f"β={beta}")
    
    axs[0].legend()
    axs[0].set_title("SmeLU")
    axs[0].grid()
    axs[1].legend()
    axs[1].set_title("SmeLU gradients")
    axs[1].grid()
    plt.tight_layout()
    # plt.savefig("example.jpg")
    plt.show()


if __name__ == '__main__':
    main()
