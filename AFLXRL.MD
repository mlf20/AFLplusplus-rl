

## Running firmwire
 Taken from the FirmWire documentation:
```
docker run --rm -it -v $(pwd):/firmwire firmwire
cd modkit/shannon && rm -rf build/ && make && cd ../..
```
Values to ensure that the fork server of AFL++ actually starts FirmWire
```
export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
AFL_FORKSRV_INIT_TMOUT=1000000
```



### Issue with Shannon based chips require snapshots to be saved as follows:

(https://github.com/FirmWire/FirmWire/issues/26)

Save the snapshot using FirmWire:
`./firmwire.py --snapshot-at 0x4157e501,mysnapshot --fuzz-triage gsm_cc  --fuzz-input ./modkit/shannon/build/fuzzers/gsm_cc.c  SM-GS955F/modem.bin`

Examples of running AFL++ on the snapshot:
`AFL_DEBUG_CHILD=1 /AFLplusplus/afl-fuzz -i modkit/shannon/fuzzers/ -o out -U -- ./firmwire.py --restore-snapshot mysnapshot  --fuzz gsm_cc --fuzz-input @@ SM-GS955F/modem.bin`


`AFL_DEBUG_CHILD=1 /AFLplusplus/afl-fuzz -t 1000 -i modkit/shannon/fuzzers/ -o out -U -- ./firmwire.py --restore-snapshot mysnapshot  --fuzz lte_rrc --fuzz-input @@ SM-GS955F/modem.bin`


To replay fuzz input run the following swapping `./crash.bin` for the fuzzing input that you want to use.
`./firmwire.py --restore-snapshot mysnapshot --fuzz-triage gsm_cc --fuzz-input ./crash.bin G970FXXU9ETLJ_G970FOXM9ETLJ_AUT/modem.bin`

## Running AFL with custom mutator:

export PYTHONPATH=`dirname /AFLplusplus/custom_mutators/rl4llms/rl4llms_mutator.py`
export AFL_PYTHON_MODULE=rl4llms_mutator
export AFL_CUSTOM_MUTATOR_ONLY

## Run AFL

From the docker container, in the `/firmwire` directory run:
`AFL_DEBUG_CHILD=1 /AFLplusplus/afl-fuzz -t 10000 -i lte_seeds/ -o out -U -- ./firmwire.py --restore-snapshot mysnapshot  --fuzz lte_rrc --fuzz-input @@ G970FXXU9ETLJ_G970FOXM9ETLJ_AUT/modem.bin`

This will save all information related to the run in directory `out`.

That should be it :)
