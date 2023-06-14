# Automatic Generation of Test Case Grammar from Programming Problem Descriptions

by
Dogyu Kim,
Aditi, and
Sang-Ki Ko

This repo provides the source code & data of our work.



## Abstract

> We aim to automatically generate a formal grammar that precisely describes the logical specification of test cases for programming problems. Generating high-quality test cases is a necessary step for evaluating the codes written by programmers as low-quality test cases easily fail to discriminate incorrect codes from correct "solutions" that exactly satisfy the requirements posed in programming problem descriptions. However, it is known to be highly technical to write a good test case from a given problem description without understanding the educational purpose of the problem and the logical restrictions specified in the description.

## Find information of problem
```
problem+info.py {Problem ID | Problem Name} [d, s, g, t, p, h]
```
> The last token means what you want to show in the information
  - d: Description
  - s: input specification
  - g: input test case grammer
  - t: test cases
  - p: public test cases
  - h: private test cases(hidden test cases)


## Reproducing the results

> Explain how (possibly with running scripts)


## Citation

```bib
@article{KimAK23,
  author =  {Dogyu Kim, Aditi, Sang-Ki Ko},
  title =   {Automatic Generation of Test Case Grammar from Programming Problem Descriptions},
  year =    {2023}
}
```

## License

All source code is made available under a BSD 3-clause license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.
