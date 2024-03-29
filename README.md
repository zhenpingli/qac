# Subword Language Model for Query Auto-Completion

This is the official github repository for [Subword Language Model for Query Auto-Completion](https://arxiv.org/abs/1909.00599) (EMNLP-IJCNLP 2019).


## Dependencies
- Python 3
- PyTorch
- SentencePiece


## Preparing Data
- Dowload original AOL query log dataset: `./get_data.sh`.
  This files will be saved in `data/aol/org` directory.
- Split this data into `{train, valid, test}.{query, uid, time}.txt` by giving name tag for the split and specifying time interval of each split.
  It will be generated in the `data/aol/<tag>` directory.
  Or, you can just run `split.sh` to use a pre-determined partition setting. 
  ```
  python split.py --tag full  --train_start "2006-03-01 00:00:00" --train_end "2006-05-18 00:00:00" \
                              --valid_start "2006-05-18 00:00:00" --valid_end "2006-05-25 00:00:00" \
                              --test_start  "2006-05-25 00:00:00" --test_end  "2006-06-01 00:00:00"
  ```
- Train [SentencePiece](https://github.com/google/sentencepiece/) models (char, bpe, and unigram): `./train_spms.sh`. 
  You may change the subword vocabulary size (default: 256).


## Training a language model
```
python train.py \
    --data_dir data/aol/full \
    --spm <spm> \               # char, bpe/<vocab-size>, or unigram/<vocab-size> 
    --sample -1 0.2 \           # if spm is ungiram
    --ninp 100 \
    --nhid 600 \
    --nlayers 1 \
    --max_seq_len 40
```  

## Generating completions using a trained language model
``` 
python generate.py \
    --gen_bsz 1 \
    --beam_size 30 \
    --branching_factor 30 \
    --retrace <R> \             # for the retrace algorithm
    --nbest <n> \               # for the n-best decoding
    --do_merge \                # for marginalization
```


## Citation
If you find this work useful, please cite:
```
@article{kim2019subword,
  title={Subword Language Model for Query Auto-Completion},
  author={Kim, Gyuwan},
  journal={arXiv preprint arXiv:1909.00599},
  year={2019}
}
```


## Contact Information
Please feel free to contact Gyuwan Kim ([gyuwan.kim@navercorp.com](mailto:gyuwan.kim@navercorp.com)) if there is any question.


## License
MIT License

```
Copyright (c) 2019-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
```

## Result

## 原文
``` 
作者的LSTM  中间对hidden 做了一堆操作大概就是 各种 sum

没有length
08/31/2021 20:57:48 -  Loading tokenizer from spm/unigram/spm.model
08/31/2021 20:57:48 -    Vocab size: 256
08/31/2021 20:57:48 -  Reading a dataset (all quries from test.query.txt)
08/31/2021 20:57:48 -    Number of test data:     1990 (seen 433, unseen 1557)
08/31/2021 20:57:48 -  Loading model from /home/subword/models/unigram
08/31/2021 20:57:52 -  Generation starts!
08/31/2021 20:59:00 -    mean decode length:  7.1
08/31/2021 20:59:00 -  67.9 s | 34.1 ms/query | 29.3 qps
08/31/2021 20:59:00 -  mrr @ 1: 0.4596 (seen) 0.1265 (unseen) 0.1990 (all)
08/31/2021 20:59:00 -  mrr @ 2: 0.5277 (seen) 0.1468 (unseen) 0.2296 (all)
08/31/2021 20:59:00 -  mrr @ 3: 0.5346 (seen) 0.1523 (unseen) 0.2355 (all)
08/31/2021 20:59:00 -  mrr @ 4: 0.5427 (seen) 0.1565 (unseen) 0.2405 (all)
08/31/2021 20:59:00 -  mrr @ 5: 0.5450 (seen) 0.1578 (unseen) 0.2420 (all)
08/31/2021 20:59:00 -  mrr @ 6: 0.5462 (seen) 0.1586 (unseen) 0.2430 (all)
08/31/2021 20:59:00 -  mrr @ 7: 0.5478 (seen) 0.1593 (unseen) 0.2438 (all)
08/31/2021 20:59:00 -  mrr @ 8: 0.5481 (seen) 0.1599 (unseen) 0.2444 (all)
08/31/2021 20:59:00 -  mrr @ 9: 0.5489 (seen) 0.1605 (unseen) 0.2450 (all)
08/31/2021 20:59:00 -  mrr @10: 0.5494 (seen) 0.1609 (unseen) 0.2454 (all)
08/31/2021 20:59:00 -   
08/31/2021 20:59:00 -  pmrr @ 1: 0.5358 (seen) 0.2704 (unseen) 0.3281 (all)
08/31/2021 20:59:00 -  pmrr @ 2: 0.6028 (seen) 0.3086 (unseen) 0.3726 (all)
08/31/2021 20:59:00 -  pmrr @ 3: 0.6097 (seen) 0.3215 (unseen) 0.3842 (all)
08/31/2021 20:59:00 -  pmrr @ 4: 0.6184 (seen) 0.3282 (unseen) 0.3913 (all)
08/31/2021 20:59:00 -  pmrr @ 5: 0.6202 (seen) 0.3320 (unseen) 0.3947 (all)
08/31/2021 20:59:00 -  pmrr @ 6: 0.6217 (seen) 0.3333 (unseen) 0.3961 (all)
08/31/2021 20:59:00 -  pmrr @ 7: 0.6234 (seen) 0.3345 (unseen) 0.3974 (all)
08/31/2021 20:59:00 -  pmrr @ 8: 0.6237 (seen) 0.3357 (unseen) 0.3983 (all)
08/31/2021 20:59:00 -  pmrr @ 9: 0.6245 (seen) 0.3365 (unseen) 0.3992 (all)
08/31/2021 20:59:00 -  pmrr @10: 0.6256 (seen) 0.3373 (unseen) 0.4000 (all)

生成的时候有length

08/31/2021 21:11:07 -  Loading tokenizer from spm/unigram/spm.model
08/31/2021 21:11:07 -    Vocab size: 256
08/31/2021 21:11:07 -  Reading a dataset (all quries from test.query.txt)
08/31/2021 21:11:07 -    Number of test data:     1990 (seen 433, unseen 1557)
08/31/2021 21:11:07 -  Loading model from /home/subword/models/unigram
08/31/2021 21:11:11 -  Generation starts!
08/31/2021 21:12:14 -    mean decode length:  7.3
08/31/2021 21:12:14 -  62.6 s | 31.4 ms/query | 31.8 qps
08/31/2021 21:12:14 -  mrr @ 1: 0.4573 (seen) 0.1259 (unseen) 0.1980 (all)
08/31/2021 21:12:14 -  mrr @ 2: 0.5323 (seen) 0.1471 (unseen) 0.2309 (all)
08/31/2021 21:12:14 -  mrr @ 3: 0.5393 (seen) 0.1544 (unseen) 0.2381 (all)
08/31/2021 21:12:14 -  mrr @ 4: 0.5462 (seen) 0.1579 (unseen) 0.2424 (all)
08/31/2021 21:12:14 -  mrr @ 5: 0.5480 (seen) 0.1592 (unseen) 0.2438 (all)
08/31/2021 21:12:14 -  mrr @ 6: 0.5496 (seen) 0.1602 (unseen) 0.2450 (all)
08/31/2021 21:12:14 -  mrr @ 7: 0.5506 (seen) 0.1608 (unseen) 0.2456 (all)
08/31/2021 21:12:14 -  mrr @ 8: 0.5517 (seen) 0.1613 (unseen) 0.2462 (all)
08/31/2021 21:12:14 -  mrr @ 9: 0.5527 (seen) 0.1620 (unseen) 0.2470 (all)
08/31/2021 21:12:14 -  mrr @10: 0.5532 (seen) 0.1622 (unseen) 0.2473 (all)
08/31/2021 21:12:14 -   
08/31/2021 21:12:14 -  pmrr @ 1: 0.5312 (seen) 0.2704 (unseen) 0.3271 (all)
08/31/2021 21:12:14 -  pmrr @ 2: 0.6051 (seen) 0.3080 (unseen) 0.3726 (all)
08/31/2021 21:12:14 -  pmrr @ 3: 0.6112 (seen) 0.3232 (unseen) 0.3858 (all)
08/31/2021 21:12:14 -  pmrr @ 4: 0.6187 (seen) 0.3293 (unseen) 0.3923 (all)
08/31/2021 21:12:14 -  pmrr @ 5: 0.6215 (seen) 0.3326 (unseen) 0.3955 (all)
08/31/2021 21:12:14 -  pmrr @ 6: 0.6234 (seen) 0.3344 (unseen) 0.3973 (all)
08/31/2021 21:12:14 -  pmrr @ 7: 0.6244 (seen) 0.3356 (unseen) 0.3985 (all)
08/31/2021 21:12:14 -  pmrr @ 8: 0.6259 (seen) 0.3368 (unseen) 0.3997 (all)
08/31/2021 21:12:14 -  pmrr @ 9: 0.6272 (seen) 0.3380 (unseen) 0.4009 (all)
08/31/2021 21:12:14 -  pmrr @10: 0.6276 (seen) 0.3387 (unseen) 0.4016 (all)


pytorch  的LSTM


08/31/2021 21:44:50 -  Loading tokenizer from spm/unigram/spm.model
08/31/2021 21:44:50 -    Vocab size: 256
08/31/2021 21:44:50 -  Reading a dataset (all quries from test.query.txt)
08/31/2021 21:44:50 -    Number of test data:     1990 (seen 433, unseen 1557)
08/31/2021 21:44:50 -  Loading model from /home/subword/models/unigram
08/31/2021 21:44:54 -  Generation starts!
08/31/2021 21:45:43 -    mean decode length:  7.3
08/31/2021 21:45:43 -  49.4 s | 24.8 ms/query | 40.3 qps
08/31/2021 21:45:43 -  mrr @ 1: 0.4503 (seen) 0.1278 (unseen) 0.1980 (all)
08/31/2021 21:45:43 -  mrr @ 2: 0.5115 (seen) 0.1455 (unseen) 0.2251 (all)
08/31/2021 21:45:43 -  mrr @ 3: 0.5216 (seen) 0.1498 (unseen) 0.2307 (all)
08/31/2021 21:45:43 -  mrr @ 4: 0.5279 (seen) 0.1533 (unseen) 0.2348 (all)
08/31/2021 21:45:43 -  mrr @ 5: 0.5325 (seen) 0.1541 (unseen) 0.2364 (all)
08/31/2021 21:45:43 -  mrr @ 6: 0.5360 (seen) 0.1552 (unseen) 0.2381 (all)
08/31/2021 21:45:43 -  mrr @ 7: 0.5370 (seen) 0.1565 (unseen) 0.2393 (all)
08/31/2021 21:45:43 -  mrr @ 8: 0.5376 (seen) 0.1572 (unseen) 0.2399 (all)
08/31/2021 21:45:43 -  mrr @ 9: 0.5388 (seen) 0.1577 (unseen) 0.2406 (all)
08/31/2021 21:45:43 -  mrr @10: 0.5395 (seen) 0.1580 (unseen) 0.2410 (all)
08/31/2021 21:45:43 -   
08/31/2021 21:45:43 -  pmrr @ 1: 0.5150 (seen) 0.2685 (unseen) 0.3221 (all)
08/31/2021 21:45:43 -  pmrr @ 2: 0.5762 (seen) 0.3019 (unseen) 0.3616 (all)
08/31/2021 21:45:43 -  pmrr @ 3: 0.5908 (seen) 0.3132 (unseen) 0.3736 (all)
08/31/2021 21:45:43 -  pmrr @ 4: 0.5972 (seen) 0.3211 (unseen) 0.3812 (all)
08/31/2021 21:45:43 -  pmrr @ 5: 0.6027 (seen) 0.3234 (unseen) 0.3842 (all)
08/31/2021 21:45:43 -  pmrr @ 6: 0.6062 (seen) 0.3264 (unseen) 0.3873 (all)
08/31/2021 21:45:43 -  pmrr @ 7: 0.6072 (seen) 0.3285 (unseen) 0.3891 (all)
08/31/2021 21:45:43 -  pmrr @ 8: 0.6078 (seen) 0.3299 (unseen) 0.3904 (all)
08/31/2021 21:45:43 -  pmrr @ 9: 0.6090 (seen) 0.3308 (unseen) 0.3913 (all)
08/31/2021 21:45:43 -  pmrr @10: 0.6097 (seen) 0.3314 (unseen) 0.3920 (all)
08/31/2021 21:45:43 -   

Process finished with exit code 0



``` 
## Transformers
``` 
10/14/2021 18:55:21 -  Loading tokenizer from spm/unigram/spm.model
10/14/2021 18:55:21 -    Vocab size: 256
10/14/2021 18:55:21 -  Reading a dataset (all quries from test.query.txt)
10/14/2021 18:55:21 -    Number of test data:     4330 (seen 1064, unseen 3266)
10/14/2021 18:55:21 -  Loading model from /home/qac/models/unigram
10/14/2021 18:55:25 -  Generation starts!
10/14/2021 19:00:39 -    mean decode length:  7.9
10/14/2021 19:00:39 -  314.3 s | 72.6 ms/query | 13.8 qps
10/14/2021 19:00:39 -  mrr @ 1: 0.5019 (seen) 0.1448 (unseen) 0.2326 (all)
10/14/2021 19:00:39 -  mrr @ 2: 0.5555 (seen) 0.1661 (unseen) 0.2618 (all)
10/14/2021 19:00:39 -  mrr @ 3: 0.5752 (seen) 0.1721 (unseen) 0.2712 (all)
10/14/2021 19:00:39 -  mrr @ 4: 0.5811 (seen) 0.1755 (unseen) 0.2752 (all)
10/14/2021 19:00:39 -  mrr @ 5: 0.5856 (seen) 0.1772 (unseen) 0.2776 (all)
10/14/2021 19:00:39 -  mrr @ 6: 0.5882 (seen) 0.1786 (unseen) 0.2792 (all)
10/14/2021 19:00:39 -  mrr @ 7: 0.5894 (seen) 0.1794 (unseen) 0.2801 (all)
10/14/2021 19:00:39 -  mrr @ 8: 0.5899 (seen) 0.1799 (unseen) 0.2807 (all)
10/14/2021 19:00:39 -  mrr @ 9: 0.5906 (seen) 0.1804 (unseen) 0.2812 (all)
10/14/2021 19:00:39 -  mrr @10: 0.5910 (seen) 0.1808 (unseen) 0.2816 (all)
10/14/2021 19:00:39 -   
10/14/2021 19:00:39 -  pmrr @ 1: 0.5451 (seen) 0.2716 (unseen) 0.3388 (all)
10/14/2021 19:00:39 -  pmrr @ 2: 0.6048 (seen) 0.3085 (unseen) 0.3813 (all)
10/14/2021 19:00:39 -  pmrr @ 3: 0.6220 (seen) 0.3219 (unseen) 0.3956 (all)
10/14/2021 19:00:39 -  pmrr @ 4: 0.6281 (seen) 0.3277 (unseen) 0.4016 (all)
10/14/2021 19:00:39 -  pmrr @ 5: 0.6334 (seen) 0.3311 (unseen) 0.4053 (all)
10/14/2021 19:00:39 -  pmrr @ 6: 0.6368 (seen) 0.3346 (unseen) 0.4088 (all)
10/14/2021 19:00:39 -  pmrr @ 7: 0.6379 (seen) 0.3359 (unseen) 0.4101 (all)
10/14/2021 19:00:39 -  pmrr @ 8: 0.6385 (seen) 0.3371 (unseen) 0.4112 (all)
10/14/2021 19:00:39 -  pmrr @ 9: 0.6393 (seen) 0.3381 (unseen) 0.4121 (all)
10/14/2021 19:00:39 -  pmrr @10: 0.6400 (seen) 0.3386 (unseen) 0.4127 (all)



config:
  "n_ctx": 1024,
  "n_embd": 512,
  "n_head": 8,
  "n_layer": 8,
  "n_positions": 1024,
  
完整的结果： 2080ti
10/18/2021 19:04:29 -    mean decode length:  8.1
10/18/2021 19:04:29 -  97905.4 s | 74.3 ms/query | 13.5 qps
10/18/2021 19:04:34 -  mrr @ 1: 0.3929 (seen) 0.1342 (unseen) 0.2659 (all)
10/18/2021 19:04:34 -  mrr @ 2: 0.4416 (seen) 0.1527 (unseen) 0.2998 (all)
10/18/2021 19:04:34 -  mrr @ 3: 0.4554 (seen) 0.1585 (unseen) 0.3097 (all)
10/18/2021 19:04:34 -  mrr @ 4: 0.4617 (seen) 0.1613 (unseen) 0.3142 (all)
10/18/2021 19:04:34 -  mrr @ 5: 0.4648 (seen) 0.1629 (unseen) 0.3166 (all)
10/18/2021 19:04:34 -  mrr @ 6: 0.4666 (seen) 0.1640 (unseen) 0.3180 (all)
10/18/2021 19:04:34 -  mrr @ 7: 0.4681 (seen) 0.1647 (unseen) 0.3192 (all)
10/18/2021 19:04:34 -  mrr @ 8: 0.4690 (seen) 0.1653 (unseen) 0.3199 (all)
10/18/2021 19:04:34 -  mrr @ 9: 0.4697 (seen) 0.1657 (unseen) 0.3205 (all)
10/18/2021 19:04:34 -  mrr @10: 0.4702 (seen) 0.1660 (unseen) 0.3209 (all)
10/18/2021 19:04:34 -   
10/18/2021 19:04:34 -  pmrr @ 1: 0.4622 (seen) 0.2976 (unseen) 0.3814 (all)
10/18/2021 19:04:34 -  pmrr @ 2: 0.5226 (seen) 0.3428 (unseen) 0.4343 (all)
10/18/2021 19:04:34 -  pmrr @ 3: 0.5400 (seen) 0.3573 (unseen) 0.4503 (all)
10/18/2021 19:04:34 -  pmrr @ 4: 0.5477 (seen) 0.3647 (unseen) 0.4579 (all)
10/18/2021 19:04:34 -  pmrr @ 5: 0.5520 (seen) 0.3690 (unseen) 0.4621 (all)
10/18/2021 19:04:34 -  pmrr @ 6: 0.5544 (seen) 0.3717 (unseen) 0.4647 (all)
10/18/2021 19:04:34 -  pmrr @ 7: 0.5564 (seen) 0.3736 (unseen) 0.4667 (all)
10/18/2021 19:04:34 -  pmrr @ 8: 0.5576 (seen) 0.3749 (unseen) 0.4679 (all)
10/18/2021 19:04:34 -  pmrr @ 9: 0.5585 (seen) 0.3759 (unseen) 0.4689 (all)
10/18/2021 19:04:34 -  pmrr @10: 0.5591 (seen) 0.3767 (unseen) 0.4696 (all)
##蒸馏的结果 2080ti

11/08/2021 23:13:23 -  45624.3 s | 34.6 ms/query | 28.9 qps
11/08/2021 23:13:29 -  mrr @ 1: 0.3496 (seen) 0.1204 (unseen) 0.2371 (all)
11/08/2021 23:13:29 -  mrr @ 2: 0.3987 (seen) 0.1368 (unseen) 0.2701 (all)
11/08/2021 23:13:29 -  mrr @ 3: 0.4102 (seen) 0.1420 (unseen) 0.2785 (all)
11/08/2021 23:13:29 -  mrr @ 4: 0.4152 (seen) 0.1446 (unseen) 0.2824 (all)
11/08/2021 23:13:29 -  mrr @ 5: 0.4182 (seen) 0.1461 (unseen) 0.2846 (all)
11/08/2021 23:13:29 -  mrr @ 6: 0.4200 (seen) 0.1471 (unseen) 0.2860 (all)
11/08/2021 23:13:29 -  mrr @ 7: 0.4213 (seen) 0.1478 (unseen) 0.2870 (all)
11/08/2021 23:13:29 -  mrr @ 8: 0.4223 (seen) 0.1484 (unseen) 0.2878 (all)
11/08/2021 23:13:29 -  mrr @ 9: 0.4229 (seen) 0.1488 (unseen) 0.2883 (all)
11/08/2021 23:13:29 -  mrr @10: 0.4234 (seen) 0.1491 (unseen) 0.2887 (all)
11/08/2021 23:13:29 -
11/08/2021 23:13:29 -  pmrr @ 1: 0.4297 (seen) 0.2868 (unseen) 0.3596 (all)
11/08/2021 23:13:29 -  pmrr @ 2: 0.4920 (seen) 0.3290 (unseen) 0.4120 (all)
11/08/2021 23:13:29 -  pmrr @ 3: 0.5078 (seen) 0.3426 (unseen) 0.4267 (all)
11/08/2021 23:13:29 -  pmrr @ 4: 0.5145 (seen) 0.3494 (unseen) 0.4334 (all)
11/08/2021 23:13:29 -  pmrr @ 5: 0.5186 (seen) 0.3532 (unseen) 0.4374 (all)
11/08/2021 23:13:29 -  pmrr @ 6: 0.5209 (seen) 0.3558 (unseen) 0.4399 (all)
11/08/2021 23:13:29 -  pmrr @ 7: 0.5228 (seen) 0.3577 (unseen) 0.4417 (all)
11/08/2021 23:13:29 -  pmrr @ 8: 0.5240 (seen) 0.3590 (unseen) 0.4430 (all)
11/08/2021 23:13:29 -  pmrr @ 9: 0.5249 (seen) 0.3600 (unseen) 0.4440 (all)
11/08/2021 23:13:29 -  pmrr @10: 0.5257 (seen) 0.3607 (unseen) 0.4447 (all)

``` 
``` 
12head 6 层 3090
11/16/2021 11:12:15 -    mean decode length:  8.8
11/16/2021 11:12:15 -  98982.9 s | 75.1 ms/query | 13.3 qps
11/16/2021 11:12:24 -  mrr @ 1: 0.4513 (seen) 0.1578 (unseen) 0.3072 (all)
11/16/2021 11:12:24 -  mrr @ 2: 0.5048 (seen) 0.1798 (unseen) 0.3453 (all)
11/16/2021 11:12:24 -  mrr @ 3: 0.5194 (seen) 0.1865 (unseen) 0.3560 (all)
11/16/2021 11:12:24 -  mrr @ 4: 0.5256 (seen) 0.1898 (unseen) 0.3607 (all)
11/16/2021 11:12:24 -  mrr @ 5: 0.5286 (seen) 0.1916 (unseen) 0.3632 (all)
11/16/2021 11:12:24 -  mrr @ 6: 0.5306 (seen) 0.1927 (unseen) 0.3648 (all)
11/16/2021 11:12:24 -  mrr @ 7: 0.5321 (seen) 0.1935 (unseen) 0.3659 (all)
11/16/2021 11:12:24 -  mrr @ 8: 0.5332 (seen) 0.1941 (unseen) 0.3667 (all)
11/16/2021 11:12:24 -  mrr @ 9: 0.5339 (seen) 0.1945 (unseen) 0.3673 (all)
11/16/2021 11:12:24 -  mrr @10: 0.5343 (seen) 0.1948 (unseen) 0.3677 (all)
11/16/2021 11:12:24 -
11/16/2021 11:12:24 -  pmrr @ 1: 0.5052 (seen) 0.3099 (unseen) 0.4093 (all)
11/16/2021 11:12:24 -  pmrr @ 2: 0.5678 (seen) 0.3585 (unseen) 0.4650 (all)
11/16/2021 11:12:24 -  pmrr @ 3: 0.5853 (seen) 0.3745 (unseen) 0.4818 (all)
11/16/2021 11:12:24 -  pmrr @ 4: 0.5929 (seen) 0.3826 (unseen) 0.4897 (all)
11/16/2021 11:12:24 -  pmrr @ 5: 0.5969 (seen) 0.3873 (unseen) 0.4940 (all)
11/16/2021 11:12:24 -  pmrr @ 6: 0.5995 (seen) 0.3903 (unseen) 0.4968 (all)
11/16/2021 11:12:24 -  pmrr @ 7: 0.6012 (seen) 0.3923 (unseen) 0.4987 (all)
11/16/2021 11:12:24 -  pmrr @ 8: 0.6025 (seen) 0.3938 (unseen) 0.5000 (all)
11/16/2021 11:12:24 -  pmrr @ 9: 0.6035 (seen) 0.3948 (unseen) 0.5010 (all)
11/16/2021 11:12:24 -  pmrr @10: 0.6040 (seen) 0.3956 (unseen) 0.5017 (all)

蒸馏12 head 2层 2080ti
11/16/2021 20:29:17 -    mean decode length:  8.2
11/16/2021 20:29:17 -  72101.1 s | 54.7 ms/query | 18.3 qps
11/16/2021 20:29:22 -  mrr @ 1: 0.4025 (seen) 0.1366 (unseen) 0.2720 (all)
11/16/2021 20:29:22 -  mrr @ 2: 0.4528 (seen) 0.1553 (unseen) 0.3068 (all)
11/16/2021 20:29:22 -  mrr @ 3: 0.4673 (seen) 0.1612 (unseen) 0.3170 (all)
11/16/2021 20:29:22 -  mrr @ 4: 0.4728 (seen) 0.1641 (unseen) 0.3213 (all)
11/16/2021 20:29:22 -  mrr @ 5: 0.4764 (seen) 0.1658 (unseen) 0.3239 (all)
11/16/2021 20:29:22 -  mrr @ 6: 0.4785 (seen) 0.1669 (unseen) 0.3255 (all)
11/16/2021 20:29:22 -  mrr @ 7: 0.4799 (seen) 0.1676 (unseen) 0.3266 (all)
11/16/2021 20:29:22 -  mrr @ 8: 0.4808 (seen) 0.1682 (unseen) 0.3273 (all)
11/16/2021 20:29:22 -  mrr @ 9: 0.4815 (seen) 0.1686 (unseen) 0.3279 (all)
11/16/2021 20:29:22 -  mrr @10: 0.4821 (seen) 0.1689 (unseen) 0.3284 (all)
11/16/2021 20:29:22 -
11/16/2021 20:29:22 -  pmrr @ 1: 0.4691 (seen) 0.2979 (unseen) 0.3851 (all)
11/16/2021 20:29:22 -  pmrr @ 2: 0.5304 (seen) 0.3433 (unseen) 0.4385 (all)
11/16/2021 20:29:22 -  pmrr @ 3: 0.5481 (seen) 0.3581 (unseen) 0.4548 (all)
11/16/2021 20:29:22 -  pmrr @ 4: 0.5555 (seen) 0.3657 (unseen) 0.4623 (all)
11/16/2021 20:29:22 -  pmrr @ 5: 0.5600 (seen) 0.3700 (unseen) 0.4667 (all)
11/16/2021 20:29:22 -  pmrr @ 6: 0.5627 (seen) 0.3727 (unseen) 0.4694 (all)
11/16/2021 20:29:22 -  pmrr @ 7: 0.5647 (seen) 0.3747 (unseen) 0.4714 (all)
11/16/2021 20:29:22 -  pmrr @ 8: 0.5658 (seen) 0.3761 (unseen) 0.4727 (all)
11/16/2021 20:29:22 -  pmrr @ 9: 0.5667 (seen) 0.3771 (unseen) 0.4737 (all)
11/16/2021 20:29:22 -  pmrr @10: 0.5675 (seen) 0.3780 (unseen) 0.4745 (all)
11/16/2021 20:29:22 -

不蒸馏12 head 2层 2080ti
11/20/2021 00:54:46 -    mean decode length:  8.5
11/20/2021 00:54:46 -  65352.9 s | 49.6 ms/query | 20.2 qps
11/20/2021 00:54:55 -  mrr @ 1: 0.4137 (seen) 0.1405 (unseen) 0.2796 (all)
11/20/2021 00:54:55 -  mrr @ 2: 0.4656 (seen) 0.1598 (unseen) 0.3155 (all)
11/20/2021 00:54:55 -  mrr @ 3: 0.4805 (seen) 0.1661 (unseen) 0.3261 (all)
11/20/2021 00:54:55 -  mrr @ 4: 0.4867 (seen) 0.1690 (unseen) 0.3308 (all)
11/20/2021 00:54:55 -  mrr @ 5: 0.4898 (seen) 0.1707 (unseen) 0.3332 (all)
11/20/2021 00:54:55 -  mrr @ 6: 0.4918 (seen) 0.1718 (unseen) 0.3347 (all)
11/20/2021 00:54:55 -  mrr @ 7: 0.4932 (seen) 0.1726 (unseen) 0.3358 (all)
11/20/2021 00:54:55 -  mrr @ 8: 0.4941 (seen) 0.1731 (unseen) 0.3365 (all)
11/20/2021 00:54:55 -  mrr @ 9: 0.4950 (seen) 0.1736 (unseen) 0.3372 (all)
11/20/2021 00:54:55 -  mrr @10: 0.4955 (seen) 0.1739 (unseen) 0.3376 (all)
11/20/2021 00:54:55 -
11/20/2021 00:54:55 -  pmrr @ 1: 0.4742 (seen) 0.2955 (unseen) 0.3865 (all)
11/20/2021 00:54:55 -  pmrr @ 2: 0.5370 (seen) 0.3412 (unseen) 0.4409 (all)
11/20/2021 00:54:55 -  pmrr @ 3: 0.5544 (seen) 0.3562 (unseen) 0.4571 (all)
11/20/2021 00:54:55 -  pmrr @ 4: 0.5626 (seen) 0.3642 (unseen) 0.4652 (all)
11/20/2021 00:54:55 -  pmrr @ 5: 0.5667 (seen) 0.3686 (unseen) 0.4695 (all)
11/20/2021 00:54:55 -  pmrr @ 6: 0.5694 (seen) 0.3716 (unseen) 0.4723 (all)
11/20/2021 00:54:55 -  pmrr @ 7: 0.5712 (seen) 0.3736 (unseen) 0.4742 (all)
11/20/2021 00:54:55 -  pmrr @ 8: 0.5724 (seen) 0.3750 (unseen) 0.4755 (all)
11/20/2021 00:54:55 -  pmrr @ 9: 0.5735 (seen) 0.3760 (unseen) 0.4765 (all)
11/20/2021 00:54:55 -  pmrr @10: 0.5741 (seen) 0.3769 (unseen) 0.4773 (all)
11/20/2021 00:54:55 -
``` 
