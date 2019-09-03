class Option(object):
    def __init__(self):
        super(Option, self).__init__()
        #  ------------ General options ----------------------------------------
        self.dataPath = "/home/dataset/"  # path for loading data set
        self.dataset = "cifar10"  # options:  cifar10 | cifar100
        self.save_path = "./output/"  # output model save path
        self.nGPU = 1  # number of GPUs to use by default
        # self.nGPU = 2  # recommend Resnext use
        # self.nGPU = 4  # recommend DARTSNet  use
        self.GPU = 0  # default gpu to use, options: range(nGPU)
        self.manualSeed = 1  # add training seed
        self.tenCrop = False

        # ------------- Data options -------------------------------------------
        self.nThreads = 8  # number of data loader threads
        self.nClasses = 10  # number of classes in the dataset

        # ---------- Optimization options --------------------------------------
        self.nEpochs = 400  # number of total epochs to train
        self.batchSize = 128  # mini-batch size
        self.momentum = 0.9  # momentum
        self.weightDecay = 1e-4  # weight decay 1e-4
        self.lr = 0.1  # initial learning rate
        self.lrPolicy = "multi_step"  # options: multi_step | linear | exp | const | step
        self.power = 1  # power for inv policy (lr_policy)
        self.step = [160, 240]  # step for linear or exp learning rate policy
        self.decayRate = 0.1  # lr decay rate
        self.endlr = 0.0001

        # ---------- Model options ---------------------------------------------
        self.netType = "PreResNet"


        # --------- AuxNet options ------------------------------------------------
        self.pivotSet = [8, 13, 18, 23]
        self.optimizerAlgorithm = "SGD"


        # --------- PreResNet options ------------------------------------------------
        if self.netType == "PreResNet":
            self.depth = 56  # resnet depth: (n-2)%6==0
        self.wideFactor = 1  # wide factor for wide-resnet

        # --------- ResNet options ------------------------------------------------
        if self.netType == "ResNet":
            self.depth = 50  # options: 18 | 34 | 50 | 101 | 152

        # --------- CifarResNeXt options ------------------------------------------------
        if self.netType == "CifarResNeXt":
            self.depth = 29
        self.cardinality = 8
        self.base_width = 64
        self.widen_factor = 4

        # --------- DARTSNet options ------------------------------------------------
        self.init_channels = 36
        self.layers = 20
        self.auxiliary = True
        self.arch = 'DARTS'

        # ---------- Resume or Retrain options ---------------------------------------------
        self.resume = None
        self.retrain = None
