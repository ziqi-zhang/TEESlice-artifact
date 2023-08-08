#!usr/bin/env python
from __future__ import print_function
from multiprocessing.pool import ThreadPool

import numpy as np
import torch

from python.basic_utils import str_hash
from python.enclave_interfaces import GlobalTensor, SecretEnum
from python.quantize_net import swalp_quantize, NamedParam, dequantize_op
from python.timer_utils import NamedTimer, NamedTimerInstance
from python.common_torch import SecretConfig, mod_on_cpu,  get_random_uniform, GlobalParam, \
    quantize, generate_unquantized_tensor, dequantize, mod_move_down
from python.linear_shares import TensorLoader, conv2d_weight_grad_op
from python.torch_utils import compare_expected_actual

np.random.seed(123)
minibatch, inChan, outChan, imgHw, filHw = 64, 128, 128, 16, 3
minibatch, inChan, outChan, imgHw, filHw = 64, 64, 3, 32, 3
xShape = [minibatch, inChan, imgHw, imgHw]
wShape = [outChan, inChan, filHw, filHw]

# s consume the dummy self
ConvOp = lambda w, x: torch.conv2d(x, w, padding=1)
MatmOp = lambda w, x: torch.mm(w, x)
TargetOp = ConvOp

AQ = get_random_uniform(SecretConfig.PrimeLimit, wShape).type(SecretConfig.dtypeForCpuOp)
A0 = torch.zeros(AQ.size()).type(SecretConfig.dtypeForCpuOp)
A1 = torch.zeros(AQ.size()).type(SecretConfig.dtypeForCpuOp)
BQ = get_random_uniform(SecretConfig.PrimeLimit, xShape).type(SecretConfig.dtypeForCpuOp)
B0 = get_random_uniform(SecretConfig.PrimeLimit, xShape).type(SecretConfig.dtypeForCpuOp)
B1 = mod_on_cpu(BQ - B0)

idealC = mod_on_cpu(TargetOp(AQ.type(torch.double), BQ.type(torch.double))).type(SecretConfig.dtypeForCpuOp)
yShape = list(idealC.size()) 
C0 = get_random_uniform(SecretConfig.PrimeLimit, yShape).type(SecretConfig.dtypeForCpuOp)
C1 = get_random_uniform(SecretConfig.PrimeLimit, yShape).type(SecretConfig.dtypeForCpuOp)
Z  = mod_on_cpu(idealC - C0 - C1)


class EnclaveInterfaceTester(TensorLoader):
    def __init__(self):
        super().__init__()
        self.Name = "SingleLayer"
        self.LayerId = str_hash(self.Name)
        self.Sid = 0

    def name_modifier(self, name):
        return self.Name + "--" + str(name)

    def init_test(self):
        print()
        GlobalTensor.init()
        self.set_eid(GlobalTensor.get_eid())

    def test_tensor(self):
        print()
        print("minibatch, inChan, outChan, imgHw, filHw = %d, %d, %d, %d, %d"
              % (minibatch, inChan, outChan, imgHw, filHw))
        print("wShape", wShape)
        print("xShape", xShape)
        print("yShape", yShape)
        print()

        NamedTimer.start("InitTensor")
        self.init_enclave_tensor("BQ", BQ.size())
        NamedTimer.end("InitTensor")
        print()

        NamedTimer.start("Preprare Decrypt")
        C0Enc = self.create_encrypt_torch(C0.size())
        NamedTimer.end("Preprare Decrypt")
        C0Rec = torch.zeros(C0.size()).type(SecretConfig.dtypeForCpuOp)

        # AES Encryption and Decryption
        NamedTimer.start("AesEncrypt")
        self.aes_encrypt(C0, C0Enc)
        NamedTimer.end("AesEncrypt")

        NamedTimer.start("AesDecrypt")
        self.aes_decrypt(C0Enc, C0Rec)
        NamedTimer.end("AesDecrypt")

        print("Error of Enc and Dec:", compare_expected_actual(C0, C0Rec))
        print()

        self.init_enclave_tensor("AQ", AQ.size())
        self.init_enclave_tensor("A0", A0.size())
        self.init_enclave_tensor("A1", A1.size())
        self.init_enclave_tensor("BQ", BQ.size())
        self.init_enclave_tensor("B0", B0.size())
        self.init_enclave_tensor("B1", B1.size())

        NamedTimer.start("SetTen")
        self.set_tensor("AQ", AQ)
        NamedTimer.end("SetTen")

        # Test the Random Generation
        NamedTimer.start("GenRandomUniform: x (A)");
        get_random_uniform(SecretConfig.PrimeLimit, xShape).type(SecretConfig.dtypeForCpuOp)
        NamedTimer.end("GenRandomUniform: x (A)");

        npAQ = AQ.numpy()
        print("PrimeLimit:", SecretConfig.PrimeLimit)
        print("Python Rand max, min, avg:", np.max(npAQ), np.min(npAQ), np.average(npAQ))

        # A0 and A1 should have the same PRG
        self.set_seed("AQ", "A0")
        self.set_seed("A0", "A0")
        self.set_seed("A1", "A0")
        self.set_seed("BQ", "B0")
        self.set_seed("B0", "B0")
        self.set_seed("B1", "B0")

        NamedTimer.start("SetTen")
        self.set_tensor("AQ", AQ)
        NamedTimer.end("SetTen")

        NamedTimer.start("GetRandom")
        # self.GetRandom("A0", A0)
        self.get_random("B0", B0)
        NamedTimer.end("GetRandom")

        self.set_tensor("BQ", BQ)
        args = [("AQ", A1, "A0"), ("BQ", B1, "B0")]
        with ThreadPool(2) as pool:
            NamedTimer.start("GetShare")
            self.get_share("AQ", A1, "A0")
            self.get_share("BQ", B1, "B0")
            NamedTimer.end("GetShare")

        BQRec = mod_on_cpu(B1 + B0)
        # MoveDown(BQ)
        print(BQ[0, 0, 0, 0])
        print(BQRec[0, 0, 0, 0])
        print("Same Seed Error:", compare_expected_actual(BQ, BQRec))
        print()

        print("Reconstruct Shared C")
        CRecon = torch.zeros(C0.size())
        self.init_enclave_tensor("C0", C0.size())
        self.init_enclave_tensor("C1", C1.size())
        self.init_enclave_tensor("Z", Z.size())
        self.init_enclave_tensor("CRecon", CRecon.size())
        self.set_tensor("C0", C0)
        self.set_tensor("C1", C1)
        self.set_tensor("Z", Z)

        self.enclave_recon("C0", "C1", "Z", "CRecon")
        NamedTimer.start("Recon")
        self.enclave_recon("C0", "C1", "Z", "CRecon")
        NamedTimer.end("Recon")

        NamedTimer.start("GetTen")
        self.get_tensor("CRecon", CRecon)
        NamedTimer.end("GetTen")

        print("C Recon Error:", compare_expected_actual(idealC, CRecon))

        NamedTimer.start("SimdRecon")
        self.enclave_add3("C0", "Z", "C1", "CRecon")
        NamedTimer.end("SimdRecon")

        self.get_tensor("CRecon", CRecon)
        npReal = CRecon.numpy()
        npReal[npReal < 0] += SecretConfig.PrimeLimit

        print("C SimRecon Error:", compare_expected_actual(idealC, CRecon))

    def test_plain_compute(self):
        print()
        with NamedTimerInstance("Time of Plain Computation"):
            TargetOp(AQ, BQ)

    def test_async_test(self):
        print()
        x_shape = [512, 64, 32, 32]
        w_shape = x_shape

        def init_set(n):
            self.init_enclave_tensor(n, w_shape)
            self.generate_cpu_tensor(n, w_shape)
            self.set_seed(n, n)

        init_set("AQ")
        init_set("BQ")
        init_set("CQ")
        init_set("DQ")
        name1, tensor1 = "AQ", self.get_cpu("AQ")
        name2, tensor2 = "BQ", self.get_cpu("BQ")
        name3, tensor3 = "CQ", self.get_cpu("CQ")
        name4, tensor4 = "DQ", self.get_cpu("DQ")
        with NamedTimerInstance("GetRandom * 4"):
            self.get_random("AQ", self.get_cpu("AQ"))
            self.get_random("BQ", self.get_cpu("BQ"))
            self.get_random("CQ", self.get_cpu("CQ"))
            self.get_random("DQ", self.get_cpu("DQ"))
        with NamedTimerInstance("GetShare * 4"):
            self.get_share("AQ", self.get_cpu("AQ"), "AQ")
            self.get_share("BQ", self.get_cpu("BQ"), "BQ")
            self.get_share("CQ", self.get_cpu("CQ"), "CQ")
            self.get_share("DQ", self.get_cpu("DQ"), "DQ")
        with NamedTimerInstance("AsyncTask"):
            self.async_task(name1, tensor1, name1,
                            name2, tensor2, name2,
                            name3, tensor3, name3,
                            name4, tensor4, name4)
        print(torch.sum(self.get_cpu("AQ")))
        print(torch.sum(self.get_cpu("BQ")))

    def test_async_task(self):
        print()
        x_shape = [512, 64, 32, 32]
        w_shape = x_shape

        def init_set(name, seed, shape):
            self.init_enclave_tensor(name, w_shape)
            self.generate_cpu_tensor(name, shape)
            self.set_seed(name, seed)

        init_set("AQ", "AQ", w_shape)
        init_set("BQ", "AQ", w_shape)

        with NamedTimerInstance("AsyncGetShare"):
            get_share_id = self.async_get_share("AQ", self.get_cpu("AQ"), "AQ")
            get_random_id = self.async_get_random("BQ", self.get_cpu("BQ"), "AQ")

        check_counter = 0
        wanted_id = [get_share_id, get_random_id]
        with NamedTimerInstance("GetStatus"):
            while len(wanted_id) > 0:
                check_counter += 1
                to_be_removed = []
                for id in wanted_id:
                    status = self.get_task_status(id)
                    if status:
                        to_be_removed.append(id)
                for id in to_be_removed:
                    wanted_id.remove(id)

        print("check_counter: ", check_counter)
        print(torch.sum(self.get_cpu("AQ") + self.get_cpu("BQ")))

    def test_fused_share(self, input_af, input_bf):
        x_shape = input_af.shape
        w_shape = input_bf.shape

        self.init_enclave_tensor("Af", x_shape)
        self.init_enclave_tensor("AQ", x_shape)
        self.init_enclave_tensor("A0", x_shape)
        self.init_enclave_tensor("A1", x_shape)
        self.init_enclave_tensor("E", x_shape)
        self.init_enclave_tensor("U", x_shape)
        self.generate_cpu_tensor("Af", x_shape)
        self.generate_cpu_tensor("A1", x_shape)
        self.generate_cpu_tensor("E", x_shape)
        self.generate_cpu_tensor("U", x_shape)
        self.generate_cpu_tensor("E_nonfused", x_shape)
        self.generate_cpu_tensor("A1_nonfused", x_shape)
        self.set_seed("Af", "A0")
        self.set_seed("Af", "U")
        self.set_seed("AQ", "A0")
        self.set_seed("AQ", "U")

        self.init_enclave_tensor("Bf", w_shape)
        self.init_enclave_tensor("BQ", w_shape)
        self.init_enclave_tensor("B0", w_shape)
        self.init_enclave_tensor("B1", w_shape)
        self.init_enclave_tensor("F", w_shape)
        self.init_enclave_tensor("V", w_shape)
        self.generate_cpu_tensor("Bf", w_shape)
        self.generate_cpu_tensor("B1", w_shape)
        self.generate_cpu_tensor("F", w_shape)
        self.generate_cpu_tensor("V", w_shape)
        self.generate_cpu_tensor("F_nonfused", w_shape)
        self.generate_cpu_tensor("B1_nonfused", w_shape)
        self.set_seed("Bf", "B0")
        self.set_seed("Bf", "V")
        self.set_seed("BQ", "B0")
        self.set_seed("BQ", "V")

        print("allocated tensors")

        def quantize_a():
            self.set_cpu("Af", input_af)
            self.transfer_cpu_to_enclave("Af")
            self.set_cpu("AQ", swalp_quantize(NamedParam("Af", self.get_cpu("Af"))))

            self.set_cpu("Bf", input_bf)
            self.transfer_cpu_to_enclave("Bf")
            self.set_cpu("BQ", swalp_quantize(NamedParam("Bf", self.get_cpu("Bf"))))

        quantize_a()
        print("Initialized tensors for fused share")
        task_ids = []
        with NamedTimerInstance("get_share x 4"):
            task_ids.append(self.async_get_share("AQ", self.get_cpu("E_nonfused"), "U"))
            task_ids.append(self.async_get_share("AQ", self.get_cpu("A1_nonfused"), "A0"))
            task_ids.append(self.async_get_share("BQ", self.get_cpu("F_nonfused"), "V"))
            task_ids.append(self.async_get_share("BQ", self.get_cpu("B1_nonfused"), "B0"))
            for id in task_ids:
                while not self.get_task_status(id):
                    pass
        task_ids = []
        with NamedTimerInstance("fused_quantize_share x 4"):
            task_ids.append(self.fused_quantize_share("Af", "E", "Af", "U", is_async=True))
            task_ids.append(self.fused_quantize_share("Af", "A1", "Af", "A0", is_async=True))
            task_ids.append(self.fused_quantize_share("Bf", "F", "Bf", "V", is_async=True))
            task_ids.append(self.fused_quantize_share("Bf", "B1", "Bf", "B0", is_async=True))
            for id in task_ids:
                while not self.get_task_status(id):
                    pass
        compare_expected_actual(self.get_cpu("E_nonfused"), self.get_cpu("E"), get_relative=True, verbose=True)
        compare_expected_actual(self.get_cpu("F_nonfused"), self.get_cpu("F"), get_relative=True, verbose=True)

        quantize_a()
        print("Initialized tensors for fused share 2")
        task_ids = []
        with NamedTimerInstance("get_share x 4"):
            task_ids.append(self.async_get_share("AQ", self.get_cpu("E_nonfused"), "U"))
            task_ids.append(self.async_get_share("AQ", self.get_cpu("A1_nonfused"), "A0"))
            task_ids.append(self.async_get_share("BQ", self.get_cpu("F_nonfused"), "V"))
            task_ids.append(self.async_get_share("BQ", self.get_cpu("B1_nonfused"), "B0"))
            for id in task_ids:
                while not self.get_task_status(id):
                    pass

        task_ids = []
        with NamedTimerInstance("fused_quantize_share2 x 2"):
            task_ids.append(self.fused_quantize_share2("Af", "A1", "E", "Af", "A0", "U", is_async=True))
            task_ids.append(self.fused_quantize_share2("Bf", "B1", "F", "Bf", "B0", "V", is_async=True))
            for id in task_ids:
                while not self.get_task_status(id):
                    pass
        compare_expected_actual(self.get_cpu("E_nonfused"), self.get_cpu("E"), get_relative=True, verbose=True)
        compare_expected_actual(self.get_cpu("F_nonfused"), self.get_cpu("F"), get_relative=True, verbose=True)
        compare_expected_actual(self.get_cpu("A1_nonfused"), self.get_cpu("A1"), get_relative=True, verbose=True)
        compare_expected_actual(self.get_cpu("B1_nonfused"), self.get_cpu("B1"), get_relative=True, verbose=True)

    def marshal_fused_share(self):
        def give_test(x, w):
            self.test_fused_share(x, w)

        x_shape = [512, 64, 32, 32]
        dtype = SecretConfig.dtypeForCpuOp

        give_test(torch.from_numpy(np.random.uniform(-18, 2.0 ** (127), size=x_shape)).type(dtype),
                  torch.from_numpy(np.random.uniform(-18, 2.0 ** (127), size=x_shape)).type(dtype),)

        give_test(get_random_uniform(SecretConfig.PrimeLimit, x_shape).type(dtype),
                  get_random_uniform(SecretConfig.PrimeLimit, x_shape).type(dtype), )


    def test_fused_recon(self):
        # test: Cf ~= deQ(C' + Ci)
        x_shape = [512, 64, 32, 32]
        w_shape = [64, 64, 3, 3]
        y_shape = [512, 64, 32, 32]

        layer_name_op = "dummyLayer"

        self.init_enclave_tensor("Af", x_shape)
        self.init_enclave_tensor("E", x_shape)
        self.generate_cpu_tensor("E", x_shape)
        self.generate_cpu_tensor("AQ", x_shape)
        self.generate_cpu_tensor("Af", x_shape)
        self.set_seed("Af", "U")

        self.set_cpu("Af", get_random_uniform(1000, x_shape).type(SecretConfig.dtypeForCpuOp))
        self.transfer_cpu_to_enclave("Af")
        self.set_cpu("AQ", swalp_quantize(NamedParam(layer_name_op + "X", self.get_cpu("Af"))))
        self.fused_quantize_share("Af", "E", "Af", "U", is_async=False)

        self.init_enclave_tensor("Bf", w_shape)
        self.init_enclave_tensor("F", w_shape)
        self.generate_cpu_tensor("F", w_shape)
        self.generate_cpu_tensor("BQ", w_shape)
        self.generate_cpu_tensor("Bf", w_shape)
        self.set_seed("Bf", "V")

        self.set_cpu("Bf", get_random_uniform(1000, w_shape).type(SecretConfig.dtypeForCpuOp))
        self.transfer_cpu_to_enclave("Bf")
        self.set_cpu("BQ", swalp_quantize(NamedParam(layer_name_op + "Y", self.get_cpu("Bf"))))
        self.fused_quantize_share("Bf", "F", "Bf", "V", is_async=False)

        self.init_enclave_tensor("Cf", y_shape)
        self.init_enclave_tensor("CQ", y_shape)
        self.init_enclave_tensor("Ci", y_shape)
        self.generate_cpu_tensor("Cf", y_shape)
        self.generate_cpu_tensor("CQ", y_shape)
        self.generate_cpu_tensor("Ci", y_shape)

        frozen_cq = get_random_uniform(1000, x_shape).type(SecretConfig.dtypeForCpuOp)
        frozen_ci = get_random_uniform(1000, x_shape).type(SecretConfig.dtypeForCpuOp)

        self.set_cpu("CQ", frozen_cq.clone().detach())
        self.set_cpu("Ci", frozen_ci.clone().detach())
        self.transfer_cpu_to_enclave("CQ")
        self.transfer_cpu_to_enclave("Ci")
        with NamedTimerInstance("Fused Recon"):
            self.fused_recon("Cf", "CQ", "Ci", "Af", "Bf")
        self.transfer_enclave_to_cpu("Cf")

        ideal_cq = mod_move_down(frozen_cq + frozen_ci)
        ideal_cf = dequantize_op(NamedParam(layer_name_op+"Z", ideal_cq), layer_name_op)
        compare_expected_actual(ideal_cf, self.get_cpu("Cf"), get_relative=False, verbose=True)


Tester = EnclaveInterfaceTester()
Tester.init_test()
# Tester.test_plain_compute()
# Tester.test_tensor()
# Tester.test_quantize_only()
# Tester.test_quantize_dequantize()
# Tester.test_quantize_plainconv2d()
# Tester.test_quantize_plain_conv2d_backward_weight()
# Tester.test_async_test()
# Tester.test_async_task()
Tester.marshal_fused_share()
# Tester.test_fused_recon()
