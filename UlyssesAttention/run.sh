#!/bin/sh

torchrun --nproc_per_node=8 ulysses_attn_test.py  > 8card.log

torchrun --nproc_per_node=4 ulysses_attn_test.py  > 4card.log


python  ulysses_fa_api.py > fa_api.log 
