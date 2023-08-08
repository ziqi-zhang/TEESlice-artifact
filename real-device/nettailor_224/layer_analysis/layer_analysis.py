import os, sys
from pdb import set_trace as st
import os.path as osp
import json

def extract_time_from_line(line):
    assert "ms" in line
    time_ms = float(line.split()[-2])
    return time_ms

def stage_percentage(stage, total):
    return round(stage/total*100, 2)


dataset, model = "CIFAR10", "alexnet"

for dataset in ["CIFAR10", "CIFAR100", "STL10", "UTKFaceRace"]:
    for model in ["alexnet", "resnet18", "vgg16_bn"]:

        name = f"{dataset}_{model}"
        path = f"{name}.txt"

        EnclaveConvTotalTime, EnclaveConvInputPreprocess, EnclaveConvOutputPreprocess, EnclaveConvForward = 0, 0, 0, 0
        GPUConvTotalTime, GPUConvInputPreprocess, GPUConvForward =  0, 0, 0
        QuantReLUTotalTime, QuantReLUInputPreprocess, QuantReLUEnclaveForward, QuantReLUOutputPreprocess = 0, 0, 0, 0
        ReLUTotalTime, ReLUInputPreprocess, ReLUEnclaveForward = 0, 0, 0
        MaxpoolForward = 0

        decoded_lines = []
        with open(path) as f:
            raw_lines = f.readlines()
            for line in raw_lines:
                if "Time for" in line:
                    decoded_lines.append(line)
                if "Batch_size" in line:
                    batch_size = int(line.split()[-1])
                elif "Num_repeat =" in line:
                    num_repeat = int(line.split()[-1])
                
            last_line = raw_lines[-1]
            end2end_time = float(last_line.split()[2][:-1])
            end2end_throughput = float(last_line.split()[-3][:-1])

        for line in decoded_lines:
            if "QuantReLULayerUnitForward" in line:
                QuantReLUTotalTime += extract_time_from_line(line)
            elif "QuantReLULayerUnitInputPreprocess" in line:
                QuantReLUInputPreprocess += extract_time_from_line(line)
            elif "QuantReLULayerEnclaveForward" in line:
                QuantReLUEnclaveForward += extract_time_from_line(line)
            elif "QuantReLULayerUnitOutputPreprocess" in line:
                QuantReLUOutputPreprocess += extract_time_from_line(line)
                
            elif "ConvLayerUnitEnvlaveForward" in line:
                EnclaveConvTotalTime += extract_time_from_line(line)
            elif "ConvLayerEnclaveInputPreprocess" in line:
                EnclaveConvInputPreprocess += extract_time_from_line(line)
            elif "ConvLayerEnclaveForward" in line:
                EnclaveConvForward += extract_time_from_line(line)
            elif "ConvLayerEnclaveOutputPreprocess" in line:
                EnclaveConvOutputPreprocess += extract_time_from_line(line)
                
            elif "ConvLayerUnitGPUForward" in line:
                GPUConvTotalTime += extract_time_from_line(line)
            elif "ConvLayerGPUInputPreprocess" in line:
                GPUConvInputPreprocess += extract_time_from_line(line)
            elif "ConvLayerGPUForward" in line:
                GPUConvForward += extract_time_from_line(line)
                
            elif "ActivationLayerUnitForward" in line and "maxpool" in line:
                MaxpoolForward += extract_time_from_line(line)
            
            elif "ActivationLayerUnitForward" in line and "relu" in line:
                ReLUTotalTime += extract_time_from_line(line)
            elif "ActivationLayerUnitInputPreprocess" in line and "relu" in line:
                ReLUInputPreprocess += extract_time_from_line(line)
            elif "ActivationLayerEnclaveForward" in line and "relu" in line:
                ReLUEnclaveForward += extract_time_from_line(line)
            
            else:
                print("Incomplete line: ", line)
                

        # last_line = raw_lines[-1]
                
        total_time  = (
            EnclaveConvInputPreprocess + EnclaveConvOutputPreprocess + EnclaveConvForward +
            GPUConvInputPreprocess + GPUConvForward +
            QuantReLUInputPreprocess + QuantReLUEnclaveForward + QuantReLUOutputPreprocess +
            ReLUTotalTime +
            MaxpoolForward
        )

        TwoReLUTotalTime = ReLUTotalTime + QuantReLUTotalTime
        TwoReLUInputPreprocessTime = ReLUInputPreprocess + QuantReLUInputPreprocess + QuantReLUOutputPreprocess
        TwoReLUForward = ReLUEnclaveForward + QuantReLUEnclaveForward


        print("TotalTime:".center(30, " "), round(total_time,2), " ms" )
        print("EnclaveConvTotalTime:".center(30, " "), round(EnclaveConvTotalTime,2), " ms (", round(EnclaveConvTotalTime/total_time*100, 2), "%)")
        print("GPUConvTotalTime:".center(30, " "), round(GPUConvTotalTime,2), " ms (", round(GPUConvTotalTime/total_time*100, 2), "%)")
        print("TwoReLUTotalTime:".center(30, " "), round(TwoReLUTotalTime,2), " ms (", round(TwoReLUTotalTime/total_time*100, 2), "%)")
        print()

        TimePerSample = total_time / (batch_size*num_repeat)
        throughput = 1000 / TimePerSample
        print("TimePerSample:".center(30, " "), round(TimePerSample,2), " ms" )
        print("Throughput:".center(30, " "), round(throughput,2) )


        print("EnclaveConvInputPreprocess:".center(30, " "), round(EnclaveConvInputPreprocess,2), " ms (", round(EnclaveConvInputPreprocess/total_time*100, 2), "%)")
        print("EnclaveConvForward:".center(30, " "), round(EnclaveConvForward,2), " ms (", round(EnclaveConvForward/total_time*100, 2), "%)")
        print("EnclaveConvOutputPreprocess:".center(30, " "), round(EnclaveConvOutputPreprocess,2), " ms (", round(EnclaveConvOutputPreprocess/total_time*100, 2), "%)")

        print("GPUConvInputPreprocess:".center(30, " "), round(GPUConvInputPreprocess,2), " ms (", round(GPUConvInputPreprocess/total_time*100, 2), "%)")
        print("GPUConvForward:".center(30, " "), round(GPUConvForward,2), " ms (", round(GPUConvForward/total_time*100, 2), "%)")

        print("TwoReLUInputPreprocessTime:".center(30, " "), round(TwoReLUInputPreprocessTime,2), " ms (", round(TwoReLUInputPreprocessTime/total_time*100, 2), "%)")
        print("TwoReLUForward:".center(30, " "), round(TwoReLUForward,2), " ms (", round(TwoReLUForward/total_time*100, 2), "%)")

        print("MaxpoolForward:".center(30, " "), round(MaxpoolForward,2), " ms (", round(MaxpoolForward/total_time*100, 2), "%)")


        print("QuantReLUInputPreprocess:".center(30, " "), round(QuantReLUInputPreprocess,2), " ms (", round(QuantReLUInputPreprocess/total_time*100, 2), "%)")
        print("QuantReLUEnclaveForward:".center(30, " "), round(QuantReLUEnclaveForward,2), " ms (", round(QuantReLUEnclaveForward/total_time*100, 2), "%)")
        print("QuantReLUOutputPreprocess:".center(30, " "), round(QuantReLUOutputPreprocess,2), " ms (", round(QuantReLUOutputPreprocess/total_time*100, 2), "%)")

        print("ReLUInputPreprocess:".center(30, " "), round(ReLUInputPreprocess,2), " ms (", round(ReLUInputPreprocess/total_time*100, 2), "%)")
        print("ReLUEnclaveForward:".center(30, " "), round(ReLUEnclaveForward,2), " ms (", round(ReLUEnclaveForward/total_time*100, 2), "%)")
        print()

        DataTransfer = (
            EnclaveConvInputPreprocess + EnclaveConvOutputPreprocess + GPUConvInputPreprocess + 
            QuantReLUInputPreprocess + QuantReLUOutputPreprocess + ReLUInputPreprocess
        )
        NonLinearForward = QuantReLUEnclaveForward + ReLUEnclaveForward + MaxpoolForward

        print("EnclaveConvForward:".center(30, " "), round(EnclaveConvForward,2), " ms (", round(EnclaveConvForward/total_time*100, 2), "%)")
        print("GPUConvForward:".center(30, " "), round(GPUConvForward,2), " ms (", round(GPUConvForward/total_time*100, 2), "%)")
        print("DataTransfer:".center(30, " "), round(DataTransfer,2), " ms (", round(DataTransfer/total_time*100, 2), "%)")
        print("NonLinearForward:".center(30, " "), round(NonLinearForward,2), " ms (", round(NonLinearForward/total_time*100, 2), "%)")


        results = {
            "end2end_time": end2end_time,
            "end2end_throughput": end2end_throughput,
            
            "TimePerSample": TimePerSample,
            "Throughput": throughput,
            
            "EnclaveConvTotalTime": round(EnclaveConvTotalTime,2),
            "EnclaveConvTotalTimePercent": stage_percentage(EnclaveConvTotalTime, total_time),
            "GPUConvTotalTime": round(GPUConvTotalTime,2),
            "GPUConvTotalTimePercent": stage_percentage(GPUConvTotalTime, total_time),
            "TwoReLUTotalTime": round(GPUConvTotalTime,2),
            "TwoReLUTotalTimePercent": stage_percentage(EnclaveConvTotalTime, total_time),
            
            
        }


        for objname in [
            "EnclaveConvForward", "GPUConvForward", "DataTransfer", "NonLinearForward"
        ]:
            results[objname] = round(eval(objname), 2)
            results[f"{objname}Percent"] = stage_percentage(eval(objname), total_time)
            
        save_path = osp.join(f"{name}.json")
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)