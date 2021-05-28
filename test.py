from data_utils import *

def main():
    fine_labels, coarse_labels = get_cifar100_labels()
    print_random_cifar100_test(fine_labels, coarse_labels)
    plt.show()


if __name__ == "__main__":
    main()
