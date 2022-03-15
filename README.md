Bell@Home
===========

The aim of this project is to demonstrate an (apparent) Bell inequality violation between two computers by leveraging *measurement dependence*.

In a Bell experiment, two devices are each given a set of questions to answer separately. Comparing the way they answer their respective questions can then provide strong conclusions about the relation existing between the two devices. For instance, if the statistics of their answers violate a so-called *Bell inequality*, one can conclude that the devices shared entanglement, a genuinely *quantum* resource.

But such a conclusion only holds if some basic requirements are met. One such requirement is that the devices should not be able to influence the choice of questions being asked. Failure to satisfy this condition allows devices to create Bell-violating answers while behaving only *classically*.

The file `bah.py` implements such a scheme. It performs a Bell experiment with devices that are allowed to bias the questions asked up to some fixed amount, and produces answers to these biased questions which violate a Bell inequality. When no bias is allowed, the violation vanishes.

Importantly, the Bell violation produced by this computer program can be obtained on two separated devices (i.e. two computer with no means of communication). This is possible because each set of questions is answered strictly independently of the other device's questions.

*Bell@Home* uses a local random bit string to derive the questions to be asked to the devices. The bias is applied on this random bit string before computing the questions: each bit in the string can be biased up to the desired amount. Since *Bell@Home* uses a Bell test with binary questions, each question can be generated with a single bit from the random string. Following [[Abellan et al. (2015)]](https://doi.org/10.1103/PhysRevLett.115.250403), questions are defined as the XOR of the previous question with a fresh random bit from the string. It is also possible to use more than one fresh bit from the random string to compute each new questions. In this case, the setting is defined as the XOR of the previous question together with `k` fresh bits from the random string.


Usage
-----

For simplicity, the script `demo.py` incorporates all the steps of a complete Bell experiment in terms of three parameters:

- The number of questions `n`.
- The maximum bias `epsilon`.
- The number of random bits per question `k` to be used when creating the questions.

The parts of this script corresponding to either Alice or Bob can be run on separated computers.


Detailed usage
-----

The file `bah.py` defines a class `BellAtHome`, to be initialized with the following parameters

- The number of questions `n`.
- The maximum bias `epsilon`. This describes how much the devices are allowed to modify the input randomness to their liking (and therefore influence the Bell score they will ``achieve``).
- The number of random bits per question `k` to be used when creating the questions.
- The name of the device: either `Alice` or `Bob`.

The initial randomness of the experiment can be provided manually in two files `randomnessAlice.dat` and `randomnessBob.dat`. These files should be in the format described below. They should contain `n*k` bits. Alternatively, it is possible to generate an initial random string with the command `device.generateRandomness()`, where `device` is an instantiation of the above class.

Once the initial randomness files are ready, the questions to be asked to the devices can be computed by calling `device.computeQuestions()`. This function will create two new files:

- `randomness{device}_biased.dat`, which contains a biased version of the initial randomness
- `questions{device}.dat`, containing the list of questions to be asked to the device

The amount of bias introduced can be checked by running the command `device.computeAverageBias()`. The result should be close to the value of `epsilon` set previously.

To ask a device to answer its questions, call `device.answerQuestions()`. At this point, the Bell experiment is done.

All of the above steps can be performed fully independently for Alice and Bob. To compare their results, copy the answers of one device onto the other one. The command `device.computeCHSH()` can then compute the Bell score. In principle, a value beyond `2` is considered a Bell violation (but remember that the devices are cheating here because they were allowed influenced the initial randomness). When the bias `epsilon` is set to zero, a Bell value larger than `2` is only possible by statistical fluctuation (i.e. by a small amount, or for small values of `N`).


Data format
-----------
Binary data is stored as plain text inside text file with the `.dat` extension. These files only contains `0`, `1` and newline characters. Each line contains 64 `0/1` characters, except possibly the last line of the file.