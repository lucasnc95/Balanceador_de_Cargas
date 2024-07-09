#include "load_balancer.h"
#include <fstream>

void checkError(cl_int error, const std::string& message) {
    if (error != CL_SUCCESS) {
        std::cerr << message << " Error code: " << error << std::endl;
        exit(EXIT_FAILURE);
    }
}

Device_Info::Device_Info(cl_device_id device) : device(device) {
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    queue = clCreateCommandQueue(context, device, 0, nullptr);

    char buffer[1024];
    size_t size;

    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(buffer), buffer, &size);
    name = std::string(buffer, size);

    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(buffer), buffer, &size);
    vendor = std::string(buffer, size);

    cl_ulong mem_size;
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, nullptr);
    memory = static_cast<uint>(mem_size / 1024 / 1024); // MB

    cl_uint cu;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
    compute_units = cu;

    cl_uint freq;
    clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(freq), &freq, nullptr);
    clock_frequency = freq; // MHz
}

void Device_Info::print() const {
    std::cout << "Device Name: " << name << std::endl;
    std::cout << "Vendor: " << vendor << std::endl;
    std::cout << "Global Memory: " << memory << " MB" << std::endl;
    std::cout << "Compute Units: " << compute_units << std::endl;
    std::cout << "Clock Frequency: " << clock_frequency << " MHz" << std::endl;
}

LoadBalancer::LoadBalancer() : kernel(nullptr), program(nullptr), ticks(nullptr), times(nullptr), new_loads(nullptr), old_loads(nullptr) {
    // CPU_WORK_GROUP_SIZE = 8;
    // GPU_WORK_GROUP_SIZE = 64;
    PRECISAO_BALANCEAMENTO = 4;
    custom_type_set = false;
}

LoadBalancer::~LoadBalancer() {
    if (kernel()) clReleaseKernel(kernel());
    if (program()) clReleaseProgram(program());
    if (ticks) delete[] ticks;
    if (times) delete[] times;
    if (new_loads) delete[] new_loads;
    if (old_loads) delete[] old_loads;
}

void LoadBalancer::initializeMPI(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);int simulation
}

void LoadBalancer::finalizeMPI() {
    MPI_Finalize();
}

void LoadBalancer::initializeOpenCLDevices() {
    cl_uint platformCount;
    clGetPlatformIDs(0, nullptr, &platformCount);

    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);

    for (const auto& platform : platforms) {
        cl_uint deviceCount;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);

        std::vector<cl_device_id> platform_devices(deviceCount);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, platform_devices.data(), nullptr);

        for (const auto& device : platform_devices) {
            devices.push_back(Device_Info(device));
        }
    }

    if (devices.empty()) {
        std::cerr << "No OpenCL devices found.\n";
        finalizeMPI();
        exit(EXIT_FAILURE);
    }

    devicesWorld.resize(world_size, 0);
    devicesWorld[world_rank] = devices.size();

    MPI_Allreduce(MPI_IN_PLACE, devicesWorld.data(), world_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    totalDevices = 0;
    for (int count = 0; count < world_size; count++) {
        if (count == world_rank) {
            myDeviceOffset = totalDevices;
            myDeviceLength = devicesWorld[count];
        }
        totalDevices += devicesWorld[count];
    }

    ticks = new long int[totalDevices];
    times = new double[totalDevices];
    new_loads = new float[totalDevices];
    old_loads = new float[totalDevices];
}

void LoadBalancer::createKernel(const std::string& source_file, const std::string& kernel_name, const std::vector<size_t>& arg_sizes, const std::vector<void*>& arg_values) {
    std::ifstream source_file_stream(source_file);
    std::string source_code((std::istreambuf_iterator<char>(source_file_stream)), std::istreambuf_iterator<char>());
    const char* source_str = source_code.c_str();
    size_t source_size = source_code.length();

    cl_int error;
    program = cl::Program(devices[0].context, source_str, true, &error);
    checkError(error, "Failed to create program");

    error = program.build(devices[0].device);
    if (error != CL_SUCCESS) {
        std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0].device);
        std::cerr << "Build log:\n" << log << std::endl;
        checkError(error, "Failed to build program");
    }

    kernel = cl::Kernel(program, kernel_name.c_str(), &error);
    checkError(error, "Failed to create kernel");

    for (size_t i = 0; i < arg_sizes.size(); i++) {
        error = kernel.setArg(i, arg_sizes[i], arg_values[i]);
        checkError(error, "Failed to set kernel argument");
    }
}

void LoadBalancer::probing(int steps, bool use_default_kernel) {
    // Implementação do probing
}

void LoadBalancer::balanceLoad() {
    precisionBalance();
}

void LoadBalancer::executeKernel() {
    // Implementação da execução do kernel com medição de tempo
}

void LoadBalancer::gatherData() {
    // Implementação da função para juntar os dados processados na memória principal
}

void LoadBalancer::setDataDivision(const std::vector<int>& custom_division) {
    // Implementação da função para configurar divisão de dados personalizada
}

void LoadBalancer::computeLoad(const long int *ticks, const float *old_loads, float *new_loads, int participants) {
    if (participants == 1) {
        new_loads[0] = 1.0f;
        return;
    }

    float totalLoad = 0.0f;
    for (int count = 0; count < participants; count++) {
        totalLoad += ((count == 0) ? (old_loads[count] - 0.0f) : (old_loads[count] - old_loads[count - 1])) * ((count == 0) ? 1.0f : ((float)ticks[0]) / ((float)ticks[count]));
    }

    for (int count = 0; count < participants; count++) {
        float newLoad = (((count == 0) ? (old_loads[count] - 0.0f) : (old_loads[count] - old_loads[count - 1])) * ((count == 0) ? 1.0f : ((float)ticks[0]) / ((float)ticks[count]))) / totalLoad;
        new_loads[count] = ((count == 0) ? newLoad : new_loads[count - 1] + newLoad);
    }
}

bool LoadBalancer::computeIntersection(int offset1, int length1, int offset2, int length2, int *intersect_offset, int *intersect_length) {
    if (offset1 + length1 <= offset2) {
        return false;
    }

    if (offset1 + length1 > offset2 + length2) {
        *intersect_offset = offset2;
        *intersect_length = length2;
    } else {
        *intersect_offset = offset2;
        *intersect_length = (offset1 + length1) - offset2;
    }
    return true;
}

int LoadBalancer::getHistogramPosition(const std::vector<int>& histogram, int index) {
    int offset = 0;
    for (int count = 0; count < histogram.size(); count++) {
        if (index >= offset && index < offset + histogram[count]) {
            return count;
        }
        offset += histogram[count];
    }
    return -1;
}

float LoadBalancer::computeStdDevPercent(const long int *ticks, int participants) {
    double mean = 0.0;
    for (int count = 0; count < participants; count++) {
        mean += (double)ticks[count];
    }
    mean /= (double)participants;

    double variance = 0.0;
    for (int count = 0; count < participants; count++) {
        variance += ((double)ticks[count] - mean) * ((double)ticks[count] - mean);
    }
    variance /= (double)participants;
    return sqrt(variance) / mean;
}

float LoadBalancer::computeNorm(const float *old_loads, const float *new_loads, int participants) {
    float result = 0.0;
    for (int count = 0; count < participants; count++) {
        result += (old_loads[count] - new_loads[count]) * (old_loads[count] - new_loads[count]);
    }
    return sqrt(result);
}

void LoadBalancer::initDeviceLengthsOffsets(unsigned int offset, unsigned int length, int count) {
    deviceOffsets[count] = offset;
    deviceLengths[count] = length;
}

void LoadBalancer::precisionBalance() {
    memset(ticks, 0, sizeof(long int) * totalDevices);
    memset(times, 0, sizeof(double) * totalDevices);
    int simulation =0;
    for (int precision = 0; precision < PRECISAO_BALANCEAMENTO; precision++) {
        for (int count = 0; count < totalDevices; count++) {
            if (count >= myDeviceOffset && count < myDeviceOffset + myDeviceLength) {
                // set kernel attributes dynamically as needed
                if ((simulation % 2) == 0) {
                    // SetKernelAttribute(count - myDeviceOffset, kernelDispositivo[count], 0, swapBufferDispositivo[count][0]);
                    // SetKernelAttribute(count - myDeviceOffset, kernelDispositivo[count], 1, swapBufferDispositivo[count][1]);
                } else {
                    // SetKernelAttribute(count - myDeviceOffset, kernelDispositivo[count], 0, swapBufferDispositivo[count][1]);
                    // SetKernelAttribute(count - myDeviceOffset, kernelDispositivo[count], 1, swapBufferDispositivo[count][0]);
                }

                // kernelEventoDispositivo[count] = RunKernel(count - myDeviceOffset, kernelDispositivo[count], offset[count], length[count], isDeviceCPU(count - myDeviceOffset) ? CPU_WORK_GROUP_SIZE : GPU_WORK_GROUP_SIZE);
            }
        }
    }

    for (int count = 0; count < totalDevices; count++) {
        if (count >= myDeviceOffset && count < myDeviceOffset + myDeviceLength) {
            // SynchronizeCommandQueue(count - myDeviceOffset);
            // ticks[count] += GetEventTaskTicks(count - myDeviceOffset, kernelEventoDispositivo[count]);
        }
    }

    long int ticks_root[totalDevices];
    MPI_Allreduce(ticks, ticks_root, totalDevices, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    memcpy(ticks, ticks_root, sizeof(long int) * totalDevices);
    computeLoad(ticks, old_loads, new_loads, totalDevices);
    for (int count = 0; count < totalDevices; count++) {
        if (count >= myDeviceOffset && count < myDeviceOffset + myDeviceLength) {
            // SynchronizeCommandQueue(count - myDeviceOffset);
            times[count] = ((float)ticks[count]) / ((float)new_loads[count]);
        }
    }
    float times_root[totalDevices];
    MPI_Allreduce(times, times_root, totalDevices, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    memcpy(times, times_root, sizeof(double) * totalDevices);
}

void LoadBalancer::balanceLoad() {
    double balanceStartTime = MPI_Wtime();
    double localTimeCB;

    for (int count = 0; count < totalDevices; count++) {
        if (count >= myDeviceOffset && count < myDeviceOffset + myDeviceLength) {
            // SynchronizeCommandQueue(count - myDeviceOffset);
            localTimeCB = new_loads[count] * times[count];
        }
    }
    MPI_Allreduce(&localTimeCB, &tempoCB, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    tempoCB *= totalDevices;

    if (latencia + computeNorm(old_loads, new_loads, totalDevices) * (writeByte + banda) + tempoCB < tempoComputacaoInterna) {
        for (int count = 0; count < totalDevices; count++) {
            if (count >= myDeviceOffset && count < myDeviceOffset + myDeviceLength) {
                int newOverlapOffset = ((count == 0 ? 0.0f : new_loads[count - 1]) * totalDevices);
                int newOverlapLength = ((count == 0 ? new_loads[count] - 0.0f : new_loads[count] - new_loads[count - 1]) * totalDevices);
                for (int count2 = 0; count2 < totalDevices; count2++) {
                    if (count > count2) {
                        if (getHistogramPosition(devicesWorld, count) != getHistogramPosition(devicesWorld, count2)) {
                            int overlap[2];
                            int target = getHistogramPosition(devicesWorld, count2);
                            // T *data = (simulation % 2) == 0 ? swapBuffer[0] : swapBuffer[1];
                            // int dataDevice = (simulation % 2) == 0 ? swapBufferDispositivo[count][0] : swapBufferDispositivo[count][1];
                            MPI_Recv(overlap, 2, MPI_INT, target, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            if (overlap[1] > 0) {
                                // ReadFromMemoryObject(count - myDeviceOffset, dataDevice, (char *)(data + overlap[0] * sizeof(T)), overlap[0] * sizeof(T), overlap[1] * sizeof(T));
                                // SynchronizeCommandQueue(count - myDeviceOffset);
                                size_t loadSize = overlap[1] * sizeof(T);
                                MPI_Send((char *)(data + overlap[0] * sizeof(T)), loadSize, custom_type_set ? mpi_custom_type : mpi_data_type, target, 0, MPI_COMM_WORLD);
                            }
                        }
                    } else if (count < count2) {
                        int oldOverlapOffset = ((count2 == 0 ? 0 : old_loads[count2 - 1]) * totalDevices);
                        int oldOverlapLength = ((count2 == 0 ? old_loads[count2] - 0.0f : old_loads[count2] - old_loads[count2 - 1]) * totalDevices);

                        int intersectionOffset;
                        int intersectionLength;

                        if ((oldOverlapOffset <= newOverlapOffset - interv_balance && computeIntersection(oldOverlapOffset, oldOverlapLength, newOverlapOffset - interv_balance, newOverlapLength + interv_balance, &intersectionOffset, &intersectionLength)) ||
                            (oldOverlapOffset > newOverlapOffset - interv_balance && computeIntersection(newOverlapOffset - interv_balance, newOverlapLength + interv_balance, oldOverlapOffset, oldOverlapLength, &intersectionOffset, &intersectionLength))) {
                            if (count2 >= myDeviceOffset && count2 < myDeviceOffset + myDeviceLength) {
                                // T *data = (simulation % 2) == 0 ? swapBuffer[0] : swapBuffer[1];
                                // int dataDevice[2] = {(simulation % 2) == 0 ? swapBufferDispositivo[count][0] : swapBufferDispositivo[count][1],
                                //                      (simulation % 2) == 0 ? swapBufferDispositivo[count2][0] : swapBufferDispositivo[count2][1]};

                                // ReadFromMemoryObject(count2 - myDeviceOffset, dataDevice[1], (char *)(data + intersectionOffset * sizeof(T)), intersectionOffset * sizeof(T), intersectionLength * sizeof(T));
                                // SynchronizeCommandQueue(count2 - myDeviceOffset);
                                // WriteToMemoryObject(count - myDeviceOffset, dataDevice[0], (char *)(data + intersectionOffset * sizeof(T)), intersectionOffset * sizeof(T), intersectionLength * sizeof(T));
                                // SynchronizeCommandQueue(count - myDeviceOffset);
                            } else {
                                if (getHistogramPosition(devicesWorld, count) != getHistogramPosition(devicesWorld, count2)) {
                                    int overlap[2] = {intersectionOffset, intersectionLength};
                                    int target = getHistogramPosition(devicesWorld, count2);
                                    // T *data = (simulation % 2) == 0 ? swapBuffer[0] : swapBuffer[1];
                                    // int dataDevice = (simulation % 2) == 0 ? swapBufferDispositivo[count][0] : swapBufferDispositivo[count][1];
                                    MPI_Send(overlap, 2, MPI_INT, target, 0, MPI_COMM_WORLD);
                                    MPI_Recv((char *)(data + overlap[0] * sizeof(T)), overlap[1] * sizeof(T), custom_type_set ? mpi_custom_type : mpi_data_type, target, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                    // WriteToMemoryObject(count - myDeviceOffset, dataDevice, (char *)(data + overlap[0] * sizeof(T)), overlap[0] * sizeof(T), overlap[1] * sizeof(T));
                                    // SynchronizeCommandQueue(count - myDeviceOffset);
                                }
                            }
                        } else {
                            if (getHistogramPosition(devicesWorld, count) != getHistogramPosition(devicesWorld, count2)) {
                                int overlap[2] = {0, 0};
                                int target = getHistogramPosition(devicesWorld, count2);
                                // T *data = (simulation % 2) == 0 ? swapBuffer[0] : swapBuffer[1];
                                MPI_Send(overlap, 2, MPI_INT, target, 0, MPI_COMM_WORLD);
                            }
                        }
                    }
                }
                deviceOffsets[count] = newOverlapOffset;
                deviceLengths[count] = newOverlapLength;
                // WriteToMemoryObject(count - myDeviceOffset, DataToKernelDispositivo[count], (char *)DataToKernel, 0, sizeof(int) * 8);
                // SynchronizeCommandQueue(count - myDeviceOffset);
            }
        }
        memcpy(old_loads, new_loads, sizeof(float) * totalDevices);

        MPI_Barrier(MPI_COMM_WORLD);
        double balanceEndTime = MPI_Wtime();
        tempoBalanceamento += balanceEndTime - balanceStartTime;
    }
}

void LoadBalancer::Probing()
{
	

	double tempoInicioProbing = MPI_Wtime();
	double localLatencia = 0, localBanda = 0;
	precisionBalance();

	// Computar novas cargas.

	for (int count = 0; count < todosDispositivos; count++)
	{
		if (count >= meusDispositivosOffset && count < meusDispositivosOffset + meusDispositivosLength)
		{
			int overlapNovoOffset = ((int)(((count == 0) ? 0.0f : cargasNovas[count - 1]) * ((float)(N_Elements))));
			int overlapNovoLength = ((int)(((count == 0) ? cargasNovas[count] - 0.0f : cargasNovas[count] - cargasNovas[count - 1]) * ((float)(N_Elements))));
			for (int count2 = 0; count2 < todosDispositivos; count2++)
			{
				if (count > count2)
				{
					// Atender requisicoes de outros processos.
					if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2))
					{
						int overlap[2];
						int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
						T *Data = ((Iteration % 2) == 0) ? swapBuffer[0] : swapBuffer[1];
						int dataDevice = ((Iteration % 2) == 0) ? swapBufferDispositivo[count][0] : swapBufferDispositivo[count][1];
						MPI_Recv(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						// Podem ocorrer requisicoes vazias.
						if (overlap[1] > 0)
						{
							ReadFromMemoryObject(count - meusDispositivosOffset, dataDevice, (char *)(Data + (overlap[0] * units_per_elements)), overlap[0] * units_per_elements * sizeof(float), overlap[1] * units_per_elements * sizeof(float));
							SynchronizeCommandQueue(count - meusDispositivosOffset);

							sizeCarga = overlap[1] * units_per_elements;

							double tempoInicioBanda = MPI_Wtime();
							MPI_Ssend(Data + (overlap[0] * units_per_elements), sizeCarga, MPI_FLOAT, alvo, 0, MPI_COMM_WORLD);
							double aux = (MPI_Wtime() - tempoInicioBanda) / sizeCarga;
							localBanda = aux > localBanda ? aux : localBanda;
						}
					}
				}
				else if (count < count2)
				{
					
					int overlapAntigoOffset = ((int)(((count2 == 0) ? 0 : cargasAntigas[count2 - 1]) * (N_Elements)));
					int overlapAntigoLength = ((int)(((count2 == 0) ? cargasAntigas[count2] - 0.0f : cargasAntigas[count2] - cargasAntigas[count2 - 1]) * (N_Elements)));

					int intersecaoOffset;
					int intersecaoLength;

					if (((overlapAntigoOffset <= overlapNovoOffset - (interv_balance)) && ComputarIntersecao(overlapAntigoOffset, overlapAntigoLength, overlapNovoOffset - (interv_balance), overlapNovoLength + (interv_balance), &intersecaoOffset, &intersecaoLength)) ||
							((overlapAntigoOffset > overlapNovoOffset - (interv_balance)) && ComputarIntersecao(overlapNovoOffset - (interv_balance), overlapNovoLength + (interv_balance), overlapAntigoOffset, overlapAntigoLength, &intersecaoOffset, &intersecaoLength)))
					{
						if (count2 >= meusDispositivosOffset && count2 < meusDispositivosOffset + meusDispositivosLength)
						{
							T *Data = ((Iteration % 2) == 0) ? swapBuffer[0] : swapBuffer[1];

							int dataDevice[2] = {((Iteration % 2) == 0) ? swapBufferDispositivo[count][0] : swapBufferDispositivo[count][1],
																		((Iteration % 2) == 0) ? swapBufferDispositivo[count2][0] : swapBufferDispositivo[count2][1]};

							ReadFromMemoryObject(count2 - meusDispositivosOffset, dataDevice[1], (char *)(Data + (intersecaoOffset * units_per_elements)), intersecaoOffset * units_per_elements * sizeof(float), intersecaoLength * units_per_elements * sizeof(float));
							SynchronizeCommandQueue(count2 - meusDispositivosOffset);

							WriteToMemoryObject(count - meusDispositivosOffset, dataDevice[0], (char *)(Data + (intersecaoOffset * units_per_elements)), intersecaoOffset * units_per_elements * sizeof(float), intersecaoLength * units_per_elements * sizeof(float));
							SynchronizeCommandQueue(count - meusDispositivosOffset);
						}
						else
						{
							// Fazer uma requisicao.
							if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2))
							{
								int overlap[2] = {intersecaoOffset, intersecaoLength};
								int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
								T *Data = ((Iteration % 2) == 0) ? swapBuffer[0] : swapBuffer[1];
								int dataDevice = ((Iteration % 2) == 0) ? swapBufferDispositivo[count][0] : swapBufferDispositivo[count][1];
								SynchronizeCommandQueue(count - meusDispositivosOffset);
								double tempoInicioLatencia = MPI_Wtime();
								MPI_Ssend(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
								double aux = (MPI_Wtime() - tempoInicioLatencia) / 2;
								localLatencia = aux > localLatencia ? aux : localLatencia;

								MPI_Recv(Data + (overlap[0] * units_per_elements), overlap[1] * units_per_elements, custom_type_set ? mpi_custom_type : mpi_data_type, alvo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

								WriteToMemoryObject(count - meusDispositivosOffset, dataDevice, (char *)(Data + (overlap[0] * units_per_elements)), overlap[0] * units_per_elements * sizeof(float), overlap[1] * units_per_elements * sizeof(float));
								SynchronizeCommandQueue(count - meusDispositivosOffset);
							}
						}
					}
					else
					{
						// Fazer uma requisicao vazia.
						if (RecuperarPosicaoHistograma(dispositivosWorld, world_size, count) != RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2))
						{
							int overlap[2] = {0, 0};
							int alvo = RecuperarPosicaoHistograma(dispositivosWorld, world_size, count2);
							T *Data = ((Iteration % 2) == 0) ? swapBuffer[0] : swapBuffer[1];
							MPI_Send(overlap, 2, MPI_INT, alvo, 0, MPI_COMM_WORLD);
						}
					}
				}
			}

			offset[count] = overlapNovoOffset;
			length[count]= overlapNovoLength;

			WriteToMemoryObject(count - meusDispositivosOffset, DataToKernelDispositivo[count], (char *)DataToKernel, 0, DataToKernel_Size);
			SynchronizeCommandQueue(count - meusDispositivosOffset);
		}
	}
	memcpy(cargasAntigas, cargasNovas, sizeof(float) * todosDispositivos);

	MPI_Allreduce(&localLatencia, &latencia, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&localBanda, &banda, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	double tempoFimProbing = MPI_Wtime();
	tempoBalanceamento += tempoFimProbing - tempoInicioProbing;
	fatorErro = tempoBalanceamento;
}


//#include "load_balancer.h"

int main(int argc, char *argv[]) {
    LoadBalancer lb;
    lb.initializeMPI(argc, argv);
    lb.initializeOpenCLDevices();

    std::vector<size_t> arg_sizes = { sizeof(int), sizeof(float) };
    int int_arg = 10;
    float float_arg = 5.5f;
    std::vector<void*> arg_values = { &int_arg, &float_arg };

    lb.createKernel("kernel.cl", "exampleKernel", arg_sizes, arg_values);

  //  lb.probing(10, true);
   // lb.balanceLoad(1);
    lb.executeKernel();
    lb.gatherData();

    lb.finalizeMPI();
    return 0;
}
