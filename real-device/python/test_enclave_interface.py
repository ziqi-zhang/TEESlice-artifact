#!usr/bin/env python
from __future__ import print_function

from collections import namedtuple, defaultdict

import pytest
import torch

from python.global_config import SecretConfig
from python.common_torch import get_random_uniform, mod_on_cpu, mod_move_down
from python.enclave_interfaces import GlobalTensor
from python.tensor_loader import tensor_loader_factory, TensorLoader
from python.torch_utils import compare_expected_actual
from python.timer_utils import NamedTimerInstance
from python.quantize_net import NamedParam, swalp_quantize, dequantize_op


def test_global_tensor_init():
    assert GlobalTensor.init() is None


def test_global_tensor_double_init():
    assert GlobalTensor.init() is None
    assert GlobalTensor.init() is None


@pytest.fixture(scope="module")
def tensor_loader():
    tensor_loader = tensor_loader_factory(0, "SingleLayer")
    yield tensor_loader


def test_init_enclave_tensor(tensor_loader):
    assert tensor_loader.init_enclave_tensor("BQ", [64, 3, 3, 3]) is None


def test_set_enclave_tensor(tensor_loader):
    shape = [64, 3, 3, 3]
    name = "test_set_enclave_tensor"
    tensor = get_random_uniform(SecretConfig.PrimeLimit, shape).type(SecretConfig.dtypeForCpuOp)
    assert tensor_loader.init_enclave_tensor(name, shape) is None
    assert tensor_loader.set_enclave_tensor(name, tensor) is None


def test_set_enclave_tensor(tensor_loader):
    shape = [32, 64, 32, 32]
    name = "test_set_enclave_tensor"
    tensor = get_random_uniform(SecretConfig.PrimeLimit, shape).type(SecretConfig.dtypeForCpuOp)
    assert tensor_loader.init_enclave_tensor(name, shape) is None
    assert tensor_loader.set_enclave_tensor(name, tensor) is None


def test_get_enclave_tensor(tensor_loader):
    shape = [512, 64, 32, 32]
    name = "test_get_enclave_tensor"
    original_tensor = get_random_uniform(SecretConfig.PrimeLimit, shape).type(SecretConfig.dtypeForCpuOp)
    retrieve_tensor = torch.zeros(shape)
    assert tensor_loader.init_enclave_tensor(name, shape) is None
    with NamedTimerInstance("test_get_enclave_tensor set_enclave_tensor"):
        tensor_loader.set_enclave_tensor(name, original_tensor)
    with NamedTimerInstance("test_get_enclave_tensor get_enclave_tensor"):
        tensor_loader.get_enclave_tensor(name, retrieve_tensor)
    compare_expected_actual(original_tensor, retrieve_tensor, get_relative=True, verbose=True)
    assert compare_expected_actual(original_tensor, retrieve_tensor) == 0


def test_set_seed(tensor_loader):
    shape = [64, 3, 3, 3]
    name = "test_set_seed"
    seed_name = name + "0"

    tensor_loader.init_enclave_tensor(name, shape)
    assert tensor_loader.set_seed(name, seed_name) is None


def test_share_use_validate_seed(tensor_loader):
    shape = [64, 3, 3, 3]
    tensor_name = "test_use_validate_seed"
    seed_name = tensor_name + "seed"
    obtain_tensor = torch.zeros(shape)

    tensor_loader.init_enclave_tensor(tensor_name, shape)
    tensor_loader.set_seed(tensor_name, seed_name)
    tensor_loader.get_share(tensor_name, obtain_tensor, seed_name)


def test_share_use_invalidate_seed(tensor_loader):
    shape = [64, 3, 3, 3]
    tensor_name = "test_share_use_invalidate_seed"
    seed_name = tensor_name + "seed"
    obtain_tensor = torch.zeros(shape)

    tensor_loader.init_enclave_tensor(tensor_name, shape)
    with pytest.raises(ValueError):
        assert tensor_loader.get_share(tensor_name, obtain_tensor, seed_name)


def test_gen_share_random_same_seed(tensor_loader):
    shape = [64, 3, 3, 3]
    name = "test_gen_share_random_same_seed"
    origin_name = name + "Q"
    random_name = name + "0"
    seed_name = random_name

    tensor_loader.init_enclave_tensor(origin_name, shape)
    tensor_loader.set_seed(origin_name, seed_name)
    tensor_loader.init_enclave_tensor(random_name, shape)
    tensor_loader.set_seed(random_name, seed_name)

    original_tensor = get_random_uniform(SecretConfig.PrimeLimit, shape).type(SecretConfig.dtypeForCpuOp)
    random_tensor = torch.zeros(shape)
    share_tensor = torch.zeros(shape)

    tensor_loader.set_enclave_tensor(origin_name, original_tensor)
    tensor_loader.get_random(random_name, random_tensor)
    tensor_loader.get_share(origin_name, share_tensor, seed_name)

    retrieve_tensor = mod_on_cpu(random_tensor + share_tensor)
    assert compare_expected_actual(original_tensor, retrieve_tensor) < 1e-6

def test_async_random_share(tensor_loader):
    shape = [512, 64, 32, 32]
    # shape = [512, 64, 16, 16]

    def packed_init(name, seed):
        tensor_loader.init_enclave_tensor(name, shape)
        tensor_loader.generate_cpu_tensor(name, shape)
        tensor_loader.set_seed(name, seed)

    def gen_names(i):
        base_name = "test_async_random_share"
        share_name = base_name + "share" + str(i)
        random_name = base_name + "random" + str(i)
        seed_name = random_name + str(i)
        return share_name, random_name, seed_name

    num_pack = 6
    stop = 4
    original_tensors = []
    for i in range(num_pack):
        share_name, random_name, seed_name = gen_names(i)
        packed_init(share_name, seed_name)
        packed_init(random_name, seed_name)
        original_tensors.append(get_random_uniform(SecretConfig.PrimeLimit, shape).type(SecretConfig.dtypeForCpuOp))
        tensor_loader.set_enclave_tensor(share_name, original_tensors[i])

    def async_func(start, end):
        task_ids = []
        for i in range(start, end):
            share_name, random_name, seed_name = gen_names(i)
            share_task_id = tensor_loader.async_get_share(share_name, tensor_loader.get_cpu(share_name), seed_name)
            random_task_id = tensor_loader.async_get_random(random_name, tensor_loader.get_cpu(random_name), seed_name)
            tensor_loader.wait_tasks([random_task_id])
            task_ids.append(share_task_id)
            # task_ids.append(random_task_id)

        tensor_loader.wait_tasks(task_ids)

    with NamedTimerInstance("test_async_random_share"):
        async_func(0, stop)
    with NamedTimerInstance("test_async_random_share"):
        async_func(stop, num_pack)

    for i in range(num_pack):
        share_name, random_name, seed_name = gen_names(i)
        retrieve_tensor = mod_on_cpu(tensor_loader.get_cpu(share_name) + tensor_loader.get_cpu(random_name))
        compare_expected_actual(original_tensors[i], retrieve_tensor, get_relative=True, verbose=True)
        # assert compare_expected_actual(original_tensors[i], retrieve_tensor) < 1e-6


def test_stochastic_quantization(tensor_loader):
    # test: E + U ~= Q(Af)
    # test: E + U = A0 + A1 ~= AQ ~= Q(Af)

    shape = [512, 64, 32, 32]
    dtype = SecretConfig.dtypeForCpuOp
    input_af = get_random_uniform(SecretConfig.PrimeLimit, shape).type(dtype)
    input_bf = get_random_uniform(SecretConfig.PrimeLimit, shape).type(dtype)

    NameSeed = namedtuple("NameSeed", ("Name", "Seed"))

    enclave_names = ["Af", "AQ", "AQ_non_fused", "A0", "A1", "E", "U"]
    enclave_names += ["Bf", "BQ", "BQ_non_fused", "B0", "B1", "F", "V"]
    cpu_names = ["Af", "AQ", "A1", "E", "U", "E_nonfused", "A1_nonfused"]
    cpu_names += ["Bf", "BQ", "B1", "F", "V", "F_nonfused", "B1_nonfused"]
    seed_pairs = [NameSeed("Af", "A0"), NameSeed("Af", "U"), NameSeed("AQ", "A0"), NameSeed("AQ", "U")]
    seed_pairs += [NameSeed("AQ_non_fused", "A0"), NameSeed("AQ_non_fused", "U")]
    seed_pairs += [NameSeed("Bf", "B0"), NameSeed("Bf", "V"), NameSeed("BQ", "B0"), NameSeed("BQ", "V")]
    seed_pairs += [NameSeed("BQ_non_fused", "B0"), NameSeed("BQ_non_fused", "V")]

    for name in enclave_names:
        tensor_loader.init_enclave_tensor(name, shape)

    for name in cpu_names:
        tensor_loader.generate_cpu_tensor(name, shape)

    for tensor_name, seed_name in seed_pairs:
        tensor_loader.set_seed(tensor_name, seed_name)

    tensor_loader.set_cpu("Af", input_af)
    tensor_loader.transfer_cpu_to_enclave("Af")
    tensor_loader.set_cpu("AQ_non_fused", swalp_quantize(NamedParam("Af", tensor_loader.get_cpu("Af"))))
    AQ_without_enclave = tensor_loader.get_cpu("AQ_non_fused").clone()

    tensor_loader.set_cpu("Bf", input_bf)
    tensor_loader.transfer_cpu_to_enclave("Bf")
    tensor_loader.set_cpu("BQ_non_fused", swalp_quantize(NamedParam("Bf", tensor_loader.get_cpu("Bf"))))
    BQ_without_enclave = tensor_loader.get_cpu("BQ_non_fused").clone()

    print("Initialized tensors for fused share")
    EnclaveCpuSeed = namedtuple("EnclaveCpuSeed", ("enclave_name", "cpu_name", "seed_name"))
    non_fused_names = [EnclaveCpuSeed("AQ_non_fused", "E_nonfused", "U"),
                       EnclaveCpuSeed("AQ_non_fused", "A1_nonfused", "A0"),
                       EnclaveCpuSeed("BQ_non_fused", "F_nonfused", "V"),
                       EnclaveCpuSeed("BQ_non_fused", "B1_nonfused", "B0"),
                       ]
    with NamedTimerInstance("get_share x 4"):
        task_ids = []
        for enclave_name, cpu_name, seed_name in non_fused_names: task_ids.append(tensor_loader.async_get_share(enclave_name,
                                                          tensor_loader.get_cpu(cpu_name),
                                                          seed_name))
        tensor_loader.wait_tasks(task_ids)
    print("Complete generation for non-fused-share")

    # SequentialFunctions = namedtuple("SequentialFunctions", ("FirstFunc", "FirstArgs", "SecondFunc", "SecondArgs"))
    EnclaveFloatQuantizeCpuSeed = namedtuple("EnclaveFloatQuantizeCpuSeed",
                                          ("enclave_float", "enclave_quantize", "cpu_name", "seed_name"))
    FunctionArgs = namedtuple("FunctionArgs", ("func", "args", "kwargs"))
    fused_names = [("Af", "AQ", ("E", "U"), ("A1", "A0")),
                   ("Bf", "BQ", ("F", "V"), ("B1", "B0")),
                   ]

    task_ids = []
    consequential_actions = {}
    task_id_to_tags = defaultdict(list)
    task_tag_counter = 0
    with NamedTimerInstance("Async Quantize + Share x 4"):
        print("Going to add async tasks")
        for enclave_float, enclave_quantize, share_seed, another_share_seed in fused_names:
            task_id = tensor_loader.quantize(enclave_float, enclave_quantize, enclave_float, is_async=True)
            task_ids.append(task_id)

            consequential_actions[task_tag_counter] = lambda : \
                tensor_loader.async_get_share(enclave_quantize, tensor_loader.get_cpu(share_seed[0]), share_seed[1])
            task_id_to_tags[task_id].append(task_tag_counter)
            task_tag_counter += 1

            consequential_actions[task_tag_counter] = lambda: \
                tensor_loader.async_get_share(enclave_quantize, tensor_loader.get_cpu(another_share_seed[0]), another_share_seed[1])
            task_id_to_tags[task_id].append(task_tag_counter)
            task_tag_counter += 1

        while len(task_ids) > 0:
            to_be_removed = []
            to_be_added = []
            for task_id in task_ids:
                if tensor_loader.get_task_status(task_id):
                    to_be_removed.append(task_id)
                    for next_task_tag in task_id_to_tags[task_id]:
                        next_action = consequential_actions[next_task_tag]
                        next_task_id = next_action()
                        to_be_added.append(next_task_id)
            for task_id in to_be_removed:
                task_ids.remove(task_id)
            task_ids += to_be_added

    tensor_loader.transfer_enclave_to_cpu("AQ")
    tensor_loader.transfer_enclave_to_cpu("BQ")
    compare_expected_actual(AQ_without_enclave, tensor_loader.get_cpu("AQ"), get_relative=True, verbose=True)
    compare_expected_actual(BQ_without_enclave, tensor_loader.get_cpu("BQ"), get_relative=True, verbose=True)
    compare_expected_actual(tensor_loader.get_cpu("E_nonfused"), tensor_loader.get_cpu("E"), get_relative=True, verbose=True)
    compare_expected_actual(tensor_loader.get_cpu("F_nonfused"), tensor_loader.get_cpu("F"), get_relative=True, verbose=True)
    assert compare_expected_actual(tensor_loader.get_cpu("E_nonfused"), tensor_loader.get_cpu("E"),
                                   get_relative=True).AvgRelDiff < 1e-2
    assert compare_expected_actual(tensor_loader.get_cpu("F_nonfused"), tensor_loader.get_cpu("F"),
                                   get_relative=True).AvgRelDiff < 1e-2


def test_fused_share(tensor_loader):
    # test: E + U ~= Q(Af)
    # test: E + U = A0 + A1 ~= AQ ~= Q(Af)

    shape = [512, 64, 16, 16]
    dtype = SecretConfig.dtypeForCpuOp
    input_af = get_random_uniform(256, shape).type(dtype)
    input_bf = get_random_uniform(256, shape).type(dtype)

    NameSeed = namedtuple("NameSeed", ("Name", "Seed"))

    enclave_names = ["Af", "AQ", "A0", "A1", "E", "U"]
    enclave_names += ["Bf", "BQ", "B0", "B1", "F", "V"]
    cpu_names = ["Af", "AQ", "A1", "E", "U", "E_nonfused", "A1_nonfused"]
    cpu_names += ["Bf", "BQ", "B1", "F", "V", "F_nonfused", "B1_nonfused"]
    seed_pairs = [NameSeed("Af", "A0"), NameSeed("Af", "U"), NameSeed("AQ", "A0"), NameSeed("AQ", "U")]
    seed_pairs += [NameSeed("Bf", "B0"), NameSeed("Bf", "V"), NameSeed("BQ", "B0"), NameSeed("BQ", "V")]

    for name in enclave_names:
        tensor_loader.init_enclave_tensor(name, shape)

    for name in cpu_names:
        tensor_loader.generate_cpu_tensor(name, shape)

    for tensor_name, seed_name in seed_pairs:
        tensor_loader.set_seed(tensor_name, seed_name)

    tensor_loader.set_cpu("Af", input_af)
    tensor_loader.transfer_cpu_to_enclave("Af")
    tensor_loader.set_cpu("AQ", swalp_quantize(NamedParam("Af", tensor_loader.get_cpu("Af"))))
    tensor_loader.transfer_cpu_to_enclave("AQ")

    tensor_loader.set_cpu("Bf", input_bf)
    tensor_loader.transfer_cpu_to_enclave("Bf")
    tensor_loader.set_cpu("BQ", swalp_quantize(NamedParam("Bf", tensor_loader.get_cpu("Bf"))))
    tensor_loader.transfer_cpu_to_enclave("BQ")

    print("Initialized tensors for fused share")
    EnclaveCpuSeed = namedtuple("EnclaveCpuSeed", ("enclave_name", "cpu_name", "seed_name"))
    non_fused_names = [EnclaveCpuSeed("AQ", "E_nonfused", "U"),
                       EnclaveCpuSeed("AQ", "A1_nonfused", "A0"),
                       EnclaveCpuSeed("BQ", "F_nonfused", "V"),
                       EnclaveCpuSeed("BQ", "B1_nonfused", "B0"),
                       ]
    with NamedTimerInstance("get_share x 4"):
        task_ids = []
        for enclave_name, cpu_name, seed_name in non_fused_names:
            task_ids.append(tensor_loader.async_get_share(enclave_name,
                                                          tensor_loader.get_cpu(cpu_name),
                                                          seed_name))
        tensor_loader.wait_tasks(task_ids)
    print("Complete generation for non-fused-share")

    fused_names = [EnclaveCpuSeed("Af", "E", "U"),
                   EnclaveCpuSeed("Af", "A1", "A0"),
                   EnclaveCpuSeed("Bf", "F", "V"),
                   EnclaveCpuSeed("Bf", "B1", "B0"), ]
    with NamedTimerInstance("fused_quantize_share x 4"):
        task_ids = []
        for enclave_name, cpu_name, seed_name in fused_names:
            task_ids.append(tensor_loader.fused_quantize_share(
                enclave_name, cpu_name, enclave_name, seed_name, is_async=True))
        tensor_loader.wait_tasks(task_ids)


    # compare_expected_actual(tensor_loader.get_cpu("AQ"), tensor_loader.get_cpu("E"), get_relative=True, verbose=True)
    # compare_expected_actual(tensor_loader.get_cpu("BQ"), tensor_loader.get_cpu("F"), get_relative=True, verbose=True)
    compare_expected_actual(tensor_loader.get_cpu("E_nonfused"), tensor_loader.get_cpu("E"), get_relative=True, verbose=True)
    compare_expected_actual(tensor_loader.get_cpu("F_nonfused"), tensor_loader.get_cpu("F"), get_relative=True, verbose=True)
    compare_expected_actual(tensor_loader.get_cpu("A1_nonfused"), tensor_loader.get_cpu("A1"), get_relative=True, verbose=True)
    compare_expected_actual(tensor_loader.get_cpu("B1_nonfused"), tensor_loader.get_cpu("B1"), get_relative=True, verbose=True)


def test_fused_recon(tensor_loader):
    # test: Cf ~= deQ(C' + Ci)
    x_shape = [64, 64, 32, 32]
    w_shape = [64, 64, 3, 3]
    y_shape = [64, 64, 32, 32]

    layer_name_op = "dummyLayer"
    tensor_loader.Name = "test_fused_recon"

    print("Initializing A")
    tensor_loader.init_enclave_tensor("Af", x_shape)
    tensor_loader.init_enclave_tensor("E", x_shape)
    tensor_loader.generate_cpu_tensor("E", x_shape)
    tensor_loader.generate_cpu_tensor("AQ", x_shape)
    tensor_loader.generate_cpu_tensor("Af", x_shape)
    tensor_loader.set_seed("Af", "U")

    tensor_loader.set_cpu("Af", get_random_uniform(1000, x_shape).type(SecretConfig.dtypeForCpuOp))
    tensor_loader.transfer_cpu_to_enclave("Af")
    tensor_loader.set_cpu("AQ", swalp_quantize(NamedParam(layer_name_op + "X", tensor_loader.get_cpu("Af"))))
    tensor_loader.fused_quantize_share("Af", "E", "Af", "U", is_async=False)

    print("Initializing B")
    tensor_loader.init_enclave_tensor("Bf", w_shape)
    tensor_loader.init_enclave_tensor("F", w_shape)
    tensor_loader.generate_cpu_tensor("F", w_shape)
    tensor_loader.generate_cpu_tensor("BQ", w_shape)
    tensor_loader.generate_cpu_tensor("Bf", w_shape)
    tensor_loader.set_seed("Bf", "V")
    print("Initialized B")

    tensor_loader.set_cpu("Bf", get_random_uniform(1000, w_shape).type(SecretConfig.dtypeForCpuOp))
    tensor_loader.transfer_cpu_to_enclave("Bf")
    tensor_loader.set_cpu("BQ", swalp_quantize(NamedParam(layer_name_op + "Y", tensor_loader.get_cpu("Bf"))))
    print("Before fused quantize for B")
    tensor_loader.fused_quantize_share("Bf", "F", "Bf", "V", is_async=False)

    print("Initializing C")
    tensor_loader.init_enclave_tensor("Cf", y_shape)
    tensor_loader.init_enclave_tensor("CQ", y_shape)
    tensor_loader.init_enclave_tensor("Ci", y_shape)
    tensor_loader.generate_cpu_tensor("Cf", y_shape)
    tensor_loader.generate_cpu_tensor("CQ", y_shape)
    tensor_loader.generate_cpu_tensor("Ci", y_shape)

    frozen_cq = get_random_uniform(1000, x_shape).type(SecretConfig.dtypeForCpuOp)
    frozen_ci = get_random_uniform(1000, x_shape).type(SecretConfig.dtypeForCpuOp)

    tensor_loader.set_cpu("CQ", frozen_cq.clone().detach())
    tensor_loader.set_cpu("Ci", frozen_ci.clone().detach())
    tensor_loader.transfer_cpu_to_enclave("CQ")
    tensor_loader.transfer_cpu_to_enclave("Ci")
    with NamedTimerInstance("Fused Recon"):
        tensor_loader.fused_recon("Cf", "CQ", "Ci", "Af", "Bf")
    tensor_loader.transfer_enclave_to_cpu("Cf")

    ideal_cq = mod_move_down(frozen_cq + frozen_ci)
    ideal_cf = dequantize_op(NamedParam(layer_name_op+"Z", ideal_cq), layer_name_op)
    assert compare_expected_actual(ideal_cf, tensor_loader.get_cpu("Cf"), get_relative=True).AvgRelDiff < 1e-3

if __name__ == "__main__":
    # test_get_enclave_tensor(tensor_loader_factory(0, "SingleLayer"))
    test_async_random_share(tensor_loader_factory(0, "SingleLayer"))
    # test_fused_share(tensor_loader_factory(0, "SingleLayer"))
    # test_async_random_share(tensor_loader_factory(0, "SingleLayer"))
    # test_fused_share(tensor_loader_factory(0, "SingleLayer"))
    # test_stochastic_quantization(tensor_loader_factory(0, "SingleLayer"))
