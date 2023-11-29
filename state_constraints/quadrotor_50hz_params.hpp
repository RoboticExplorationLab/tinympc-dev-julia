#pragma once

#include <tinympc/types.hpp>

tinytype rho_value = 5.0;

tinytype Adyn_data[NSTATES*NSTATES] = {
  1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0039240,	-0.0000000,	0.0200000,	0.0000000,	0.0000000,	0.0000000,	0.0000131,	-0.0000000,	
  0.0000000,	1.0000000,	0.0000000,	-0.0039240,	0.0000000,	-0.0000000,	0.0000000,	0.0200000,	0.0000000,	-0.0000131,	0.0000000,	-0.0000000,	
  0.0000000,	0.0000000,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0200000,	0.0000000,	0.0000000,	0.0000000,	
  0.0000000,	0.0000000,	0.0000000,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0100000,	-0.0000000,	0.0000000,	
  0.0000000,	0.0000000,	0.0000000,	-0.0000000,	1.0000000,	-0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0100000,	-0.0000000,	
  0.0000000,	0.0000000,	0.0000000,	-0.0000000,	0.0000000,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.0000000,	0.0000000,	0.0100000,	
  0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.3924000,	-0.0000000,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0019620,	-0.0000000,	
  0.0000000,	0.0000000,	0.0000000,	-0.3924000,	0.0000000,	-0.0000000,	0.0000000,	1.0000000,	0.0000000,	-0.0019620,	0.0000000,	-0.0000000,	
  0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	1.0000000,	0.0000000,	0.0000000,	0.0000000,	
  0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	1.0000000,	-0.0000000,	0.0000000,	
  0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	1.0000000,	-0.0000000,	
  0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	0.0000000,	-0.0000000,	0.0000000,	1.0000000	
};

tinytype Bdyn_data[NSTATES*NINPUTS] = {
  -0.0000181,	0.0000199,	0.0000182,	-0.0000200,	
  0.0000180,	0.0000198,	-0.0000180,	-0.0000198,	
  0.0008409,	0.0008409,	0.0008409,	0.0008409,	
  -0.0275355,	-0.0303234,	0.0275663,	0.0302926,	
  -0.0276707,	0.0304278,	0.0277570,	-0.0305141,	
  0.0019748,	-0.0007224,	-0.0027844,	0.0015320,	
  -0.0036193,	0.0039800,	0.0036306,	-0.0039912,	
  0.0036016,	0.0039663,	-0.0036057,	-0.0039623,	
  0.0840857,	0.0840857,	0.0840857,	0.0840857,	
  -5.5070921,	-6.0646807,	5.5132527,	6.0585201,	
  -5.5341404,	6.0855684,	5.5513900,	-6.1028180,	
  0.3949542,	-0.1444728,	-0.5568752,	0.3063938	
};

tinytype Kinf_data[NINPUTS*NSTATES] = {
  -0.0710746,	0.0713292,	0.1004177,	-0.3163894,	-0.3243431,	-0.1978173,	-0.0521224,	0.0520576,	0.1131664,	-0.0276501,	-0.0318484,	-0.1152998,	
  0.0691497,	0.0695143,	0.1004177,	-0.2937420,	0.3128609,	0.1975443,	0.0505056,	0.0502159,	0.1131664,	-0.0209056,	0.0307243,	0.1149232,	
  0.0713801,	-0.0714524,	0.1004177,	0.3052653,	0.2965127,	-0.1968591,	0.0515793,	-0.0518414,	0.1131664,	0.0220319,	0.0178007,	-0.1139967,	
  -0.0694552,	-0.0693911,	0.1004177,	0.3048661,	-0.2850304,	0.1971321,	-0.0499624,	-0.0504322,	0.1131664,	0.0265238,	-0.0166766,	0.1143733	
};

tinytype Pinf_data[NSTATES*NSTATES] = {
  3813.5894818,	-8.1294045,	0.0000000,	32.3510220,	3398.2365637,	39.2628444,	1122.0003423,	-5.8363763,	0.0000000,	1.9851490,	46.5204264,	-1.2381469,	
  -8.1294045,	3812.1702300,	-0.0000000,	-3394.2027051,	-32.3983156,	-15.7097225,	-5.8396639,	1120.9518152,	-0.0000000,	-46.8145880,	-1.9907095,	0.4693422,	
  0.0000000,	-0.0000000,	5916.5184925,	0.0000000,	0.0000000,	-0.0000000,	0.0000000,	-0.0000000,	3049.6627578,	-0.0000000,	-0.0000000,	-0.0000000,	
  32.3510220,	-3394.2027051,	0.0000000,	12914.7580469,	181.5775672,	575.0075694,	25.5727583,	-2336.5220506,	-0.0000000,	199.9161500,	20.6720751,	182.2863984,	
  3398.2365637,	-32.3983156,	0.0000000,	181.5775672,	13016.8603134,	1437.4394210,	2343.2581524,	-25.5932679,	-0.0000000,	20.6626860,	222.6459854,	455.4430665,	
  39.2628444,	-15.7097225,	-0.0000000,	575.0075694,	1437.4394210,	23529.5557543,	75.8574104,	-30.3465164,	-0.0000000,	154.0939221,	385.2193714,	6267.4997714,	
  1122.0003423,	-5.8396639,	0.0000000,	25.5727583,	2343.2581524,	75.8574104,	635.2522411,	-4.3268743,	0.0000000,	1.7928816,	34.0204897,	11.2643601,	
  -5.8363763,	1120.9518152,	-0.0000000,	-2336.5220506,	-25.5932679,	-30.3465164,	-4.3268743,	634.2801037,	-0.0000000,	-33.5912697,	-1.7959460,	-4.5242300,	
  0.0000000,	-0.0000000,	3049.6627578,	0.0000000,	0.0000000,	-0.0000000,	0.0000000,	-0.0000000,	3410.8390439,	-0.0000000,	-0.0000000,	-0.0000000,	
  1.9851490,	-46.8145880,	-0.0000000,	199.9161500,	20.6626860,	154.0939221,	1.7928816,	-33.5912697,	-0.0000000,	20.7288989,	6.2539792,	88.8271871,	
  46.5204264,	-1.9907095,	-0.0000000,	20.6720751,	222.6459854,	385.2193714,	34.0204897,	-1.7959460,	-0.0000000,	6.2539792,	32.1352684,	222.0423485,	
  -1.2381469,	0.4693422,	-0.0000000,	182.2863984,	455.4430665,	6267.4997714,	11.2643601,	-4.5242300,	-0.0000000,	88.8271871,	222.0423485,	3614.4286944	
};

tinytype Quu_inv_data[NINPUTS*NINPUTS] = {
  0.0002892,	-0.0000000,	0.0000953,	-0.0000004,	
  -0.0000000,	0.0002840,	-0.0000003,	0.0001005,	
  0.0000953,	-0.0000003,	0.0002889,	0.0000002,	
  -0.0000004,	0.0001005,	0.0000002,	0.0002838	
};

tinytype AmBKt_data[NSTATES*NSTATES] = {
  0.9999947,	-0.0000002,	0.0000000,	0.0002761,	-0.0081714,	0.0004955,	-0.0010688,	-0.0000361,	-0.0000000,	0.0552157,	-1.6342826,	0.0990919,	
  -0.0000002,	0.9999947,	0.0000000,	0.0081437,	-0.0002755,	-0.0001833,	-0.0000360,	-0.0010652,	0.0000000,	1.6287403,	-0.0551092,	-0.0366579,	
  -0.0000000,	0.0000000,	0.9996623,	0.0000000,	0.0000000,	-0.0000000,	0.0000000,	0.0000000,	-0.0337748,	0.0000000,	0.0000000,	-0.0000000,	
  0.0000007,	-0.0039009,	-0.0000000,	0.9647306,	0.0010127,	0.0007955,	0.0001325,	-0.3877868,	-0.0000000,	-7.0538789,	0.2025399,	0.1591072,	
  0.0039008,	-0.0000007,	-0.0000000,	0.0010166,	0.9645778,	0.0021288,	0.3877668,	-0.0001330,	-0.0000000,	0.2033270,	-7.0844430,	0.4257527,	
  -0.0000000,	0.0000000,	-0.0000000,	-0.0000017,	-0.0000051,	0.9996832,	-0.0000007,	0.0000002,	-0.0000000,	-0.0003497,	-0.0010155,	-0.0633574,	
  0.0199961,	-0.0000001,	-0.0000000,	0.0001879,	-0.0059353,	0.0003596,	0.9992237,	-0.0000246,	-0.0000000,	0.0375858,	-1.1870563,	0.0719141,	
  -0.0000001,	0.0199961,	0.0000000,	0.0059129,	-0.0001874,	-0.0001336,	-0.0000245,	0.9992266,	0.0000000,	1.1825882,	-0.0374849,	-0.0267226,	
  -0.0000000,	0.0000000,	0.0196194,	0.0000000,	0.0000000,	0.0000000,	-0.0000000,	-0.0000000,	0.9619373,	0.0000000,	-0.0000000,	-0.0000000,	
  0.0000000,	-0.0000112,	-0.0000000,	0.0071939,	0.0000688,	0.0000602,	0.0000090,	-0.0015950,	-0.0000000,	0.4387801,	0.0137652,	0.0120425,	
  0.0000112,	-0.0000000,	-0.0000000,	0.0000692,	0.0071809,	0.0001602,	0.0015933,	-0.0000090,	-0.0000000,	0.0138364,	0.4361781,	0.0320399,	
  -0.0000000,	0.0000000,	-0.0000000,	-0.0000122,	-0.0000331,	0.0098181,	-0.0000043,	0.0000016,	-0.0000000,	-0.0024336,	-0.0066186,	0.9636162	
};

tinytype coeff_d2p_data[NSTATES*NINPUTS] = {
  0.0000000,	-0.0000000,	0.0000000,	-0.0000000,	
  -0.0000000,	0.0000000,	-0.0000000,	0.0000000,	
  -0.0000000,	-0.0000000,	-0.0000000,	-0.0000000,	
  -0.0000000,	0.0000000,	-0.0000000,	0.0000000,	
  -0.0000000,	0.0000000,	-0.0000000,	0.0000000,	
  -0.0000002,	0.0000002,	-0.0000002,	0.0000002,	
  0.0000000,	-0.0000000,	0.0000000,	-0.0000000,	
  -0.0000000,	0.0000000,	-0.0000000,	0.0000000,	
  -0.0000001,	-0.0000001,	-0.0000001,	-0.0000001,	
  0.0000000,	-0.0000000,	0.0000000,	-0.0000000,	
  0.0000000,	-0.0000000,	0.0000000,	-0.0000000,	
  0.0000002,	-0.0000002,	0.0000002,	-0.0000002	
};

tinytype Q_data[NSTATES]= {100.0000000,	100.0000000,	100.0000000,	4.0000000,	4.0000000,	400.0000000,	4.0000000,	4.0000000,	4.0000000,	2.0408163,	2.0408163,	4.0000000};

tinytype Qf_data[NSTATES]= {100.0000000,	100.0000000,	100.0000000,	4.0000000,	4.0000000,	400.0000000,	4.0000000,	4.0000000,	4.0000000,	2.0408163,	2.0408163,	4.0000000};

tinytype R_data[NINPUTS]= {2500.0000000,	2500.0000000,	2500.0000000,	2500.0000000};
