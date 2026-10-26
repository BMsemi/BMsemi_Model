module PE (
    input clk,
    input rst,

    // 28x28 image flattened -> 784 pixels
    input [7:0] in_vector_0,
    input [7:0] in_vector_1,
    input [7:0] in_vector_2,
    input [7:0] in_vector_3,
    input [7:0] in_vector_4,
    input [7:0] in_vector_5,
    input [7:0] in_vector_6,
    input [7:0] in_vector_7,
    input [7:0] in_vector_8,
    input [7:0] in_vector_9,
    input [7:0] in_vector_10,
    input [7:0] in_vector_11,
    input [7:0] in_vector_12,
    input [7:0] in_vector_13,
    input [7:0] in_vector_14,
    input [7:0] in_vector_15,
    input [7:0] in_vector_16,
    input [7:0] in_vector_17,
    input [7:0] in_vector_18,
    input [7:0] in_vector_19,
    input [7:0] in_vector_20,
    input [7:0] in_vector_21,
    input [7:0] in_vector_22,
    input [7:0] in_vector_23,
    input [7:0] in_vector_24,
    input [7:0] in_vector_25,
    input [7:0] in_vector_26,
    input [7:0] in_vector_27,
    input [7:0] in_vector_28,
    input [7:0] in_vector_29,
    input [7:0] in_vector_30,
    input [7:0] in_vector_31,
    input [7:0] in_vector_32,
    input [7:0] in_vector_33,
    input [7:0] in_vector_34,
    input [7:0] in_vector_35,
    input [7:0] in_vector_36,
    input [7:0] in_vector_37,
    input [7:0] in_vector_38,
    input [7:0] in_vector_39,
    input [7:0] in_vector_40,
    input [7:0] in_vector_41,
    input [7:0] in_vector_42,
    input [7:0] in_vector_43,
    input [7:0] in_vector_44,
    input [7:0] in_vector_45,
    input [7:0] in_vector_46,
    input [7:0] in_vector_47,
    input [7:0] in_vector_48,
    input [7:0] in_vector_49,
    input [7:0] in_vector_50,
    input [7:0] in_vector_51,
    input [7:0] in_vector_52,
    input [7:0] in_vector_53,
    input [7:0] in_vector_54,
    input [7:0] in_vector_55,
    input [7:0] in_vector_56,
    input [7:0] in_vector_57,
    input [7:0] in_vector_58,
    input [7:0] in_vector_59,
    input [7:0] in_vector_60,
    input [7:0] in_vector_61,
    input [7:0] in_vector_62,
    input [7:0] in_vector_63,
    input [7:0] in_vector_64,
    input [7:0] in_vector_65,
    input [7:0] in_vector_66,
    input [7:0] in_vector_67,
    input [7:0] in_vector_68,
    input [7:0] in_vector_69,
    input [7:0] in_vector_70,
    input [7:0] in_vector_71,
    input [7:0] in_vector_72,
    input [7:0] in_vector_73,
    input [7:0] in_vector_74,
    input [7:0] in_vector_75,
    input [7:0] in_vector_76,
    input [7:0] in_vector_77,
    input [7:0] in_vector_78,
    input [7:0] in_vector_79,
    input [7:0] in_vector_80,
    input [7:0] in_vector_81,
    input [7:0] in_vector_82,
    input [7:0] in_vector_83,
    input [7:0] in_vector_84,
    input [7:0] in_vector_85,
    input [7:0] in_vector_86,
    input [7:0] in_vector_87,
    input [7:0] in_vector_88,
    input [7:0] in_vector_89,
    input [7:0] in_vector_90,
    input [7:0] in_vector_91,
    input [7:0] in_vector_92,
    input [7:0] in_vector_93,
    input [7:0] in_vector_94,
    input [7:0] in_vector_95,
    input [7:0] in_vector_96,
    input [7:0] in_vector_97,
    input [7:0] in_vector_98,
    input [7:0] in_vector_99,
    input [7:0] in_vector_100,
    input [7:0] in_vector_101,
    input [7:0] in_vector_102,
    input [7:0] in_vector_103,
    input [7:0] in_vector_104,
    input [7:0] in_vector_105,
    input [7:0] in_vector_106,
    input [7:0] in_vector_107,
    input [7:0] in_vector_108,
    input [7:0] in_vector_109,
    input [7:0] in_vector_110,
    input [7:0] in_vector_111,
    input [7:0] in_vector_112,
    input [7:0] in_vector_113,
    input [7:0] in_vector_114,
    input [7:0] in_vector_115,
    input [7:0] in_vector_116,
    input [7:0] in_vector_117,
    input [7:0] in_vector_118,
    input [7:0] in_vector_119,
    input [7:0] in_vector_120,
    input [7:0] in_vector_121,
    input [7:0] in_vector_122,
    input [7:0] in_vector_123,
    input [7:0] in_vector_124,
    input [7:0] in_vector_125,
    input [7:0] in_vector_126,
    input [7:0] in_vector_127,
    input [7:0] in_vector_128,
    input [7:0] in_vector_129,
    input [7:0] in_vector_130,
    input [7:0] in_vector_131,
    input [7:0] in_vector_132,
    input [7:0] in_vector_133,
    input [7:0] in_vector_134,
    input [7:0] in_vector_135,
    input [7:0] in_vector_136,
    input [7:0] in_vector_137,
    input [7:0] in_vector_138,
    input [7:0] in_vector_139,
    input [7:0] in_vector_140,
    input [7:0] in_vector_141,
    input [7:0] in_vector_142,
    input [7:0] in_vector_143,
    input [7:0] in_vector_144,
    input [7:0] in_vector_145,
    input [7:0] in_vector_146,
    input [7:0] in_vector_147,
    input [7:0] in_vector_148,
    input [7:0] in_vector_149,
    input [7:0] in_vector_150,
    input [7:0] in_vector_151,
    input [7:0] in_vector_152,
    input [7:0] in_vector_153,
    input [7:0] in_vector_154,
    input [7:0] in_vector_155,
    input [7:0] in_vector_156,
    input [7:0] in_vector_157,
    input [7:0] in_vector_158,
    input [7:0] in_vector_159,
    input [7:0] in_vector_160,
    input [7:0] in_vector_161,
    input [7:0] in_vector_162,
    input [7:0] in_vector_163,
    input [7:0] in_vector_164,
    input [7:0] in_vector_165,
    input [7:0] in_vector_166,
    input [7:0] in_vector_167,
    input [7:0] in_vector_168,
    input [7:0] in_vector_169,
    input [7:0] in_vector_170,
    input [7:0] in_vector_171,
    input [7:0] in_vector_172,
    input [7:0] in_vector_173,
    input [7:0] in_vector_174,
    input [7:0] in_vector_175,
    input [7:0] in_vector_176,
    input [7:0] in_vector_177,
    input [7:0] in_vector_178,
    input [7:0] in_vector_179,
    input [7:0] in_vector_180,
    input [7:0] in_vector_181,
    input [7:0] in_vector_182,
    input [7:0] in_vector_183,
    input [7:0] in_vector_184,
    input [7:0] in_vector_185,
    input [7:0] in_vector_186,
    input [7:0] in_vector_187,
    input [7:0] in_vector_188,
    input [7:0] in_vector_189,
    input [7:0] in_vector_190,
    input [7:0] in_vector_191,
    input [7:0] in_vector_192,
    input [7:0] in_vector_193,
    input [7:0] in_vector_194,
    input [7:0] in_vector_195,
    input [7:0] in_vector_196,
    input [7:0] in_vector_197,
    input [7:0] in_vector_198,
    input [7:0] in_vector_199,
    input [7:0] in_vector_200,
    input [7:0] in_vector_201,
    input [7:0] in_vector_202,
    input [7:0] in_vector_203,
    input [7:0] in_vector_204,
    input [7:0] in_vector_205,
    input [7:0] in_vector_206,
    input [7:0] in_vector_207,
    input [7:0] in_vector_208,
    input [7:0] in_vector_209,
    input [7:0] in_vector_210,
    input [7:0] in_vector_211,
    input [7:0] in_vector_212,
    input [7:0] in_vector_213,
    input [7:0] in_vector_214,
    input [7:0] in_vector_215,
    input [7:0] in_vector_216,
    input [7:0] in_vector_217,
    input [7:0] in_vector_218,
    input [7:0] in_vector_219,
    input [7:0] in_vector_220,
    input [7:0] in_vector_221,
    input [7:0] in_vector_222,
    input [7:0] in_vector_223,
    input [7:0] in_vector_224,
    input [7:0] in_vector_225,
    input [7:0] in_vector_226,
    input [7:0] in_vector_227,
    input [7:0] in_vector_228,
    input [7:0] in_vector_229,
    input [7:0] in_vector_230,
    input [7:0] in_vector_231,
    input [7:0] in_vector_232,
    input [7:0] in_vector_233,
    input [7:0] in_vector_234,
    input [7:0] in_vector_235,
    input [7:0] in_vector_236,
    input [7:0] in_vector_237,
    input [7:0] in_vector_238,
    input [7:0] in_vector_239,
    input [7:0] in_vector_240,
    input [7:0] in_vector_241,
    input [7:0] in_vector_242,
    input [7:0] in_vector_243,
    input [7:0] in_vector_244,
    input [7:0] in_vector_245,
    input [7:0] in_vector_246,
    input [7:0] in_vector_247,
    input [7:0] in_vector_248,
    input [7:0] in_vector_249,
    input [7:0] in_vector_250,
    input [7:0] in_vector_251,
    input [7:0] in_vector_252,
    input [7:0] in_vector_253,
    input [7:0] in_vector_254,
    input [7:0] in_vector_255,
    input [7:0] in_vector_256,
    input [7:0] in_vector_257,
    input [7:0] in_vector_258,
    input [7:0] in_vector_259,
    input [7:0] in_vector_260,
    input [7:0] in_vector_261,
    input [7:0] in_vector_262,
    input [7:0] in_vector_263,
    input [7:0] in_vector_264,
    input [7:0] in_vector_265,
    input [7:0] in_vector_266,
    input [7:0] in_vector_267,
    input [7:0] in_vector_268,
    input [7:0] in_vector_269,
    input [7:0] in_vector_270,
    input [7:0] in_vector_271,
    input [7:0] in_vector_272,
    input [7:0] in_vector_273,
    input [7:0] in_vector_274,
    input [7:0] in_vector_275,
    input [7:0] in_vector_276,
    input [7:0] in_vector_277,
    input [7:0] in_vector_278,
    input [7:0] in_vector_279,
    input [7:0] in_vector_280,
    input [7:0] in_vector_281,
    input [7:0] in_vector_282,
    input [7:0] in_vector_283,
    input [7:0] in_vector_284,
    input [7:0] in_vector_285,
    input [7:0] in_vector_286,
    input [7:0] in_vector_287,
    input [7:0] in_vector_288,
    input [7:0] in_vector_289,
    input [7:0] in_vector_290,
    input [7:0] in_vector_291,
    input [7:0] in_vector_292,
    input [7:0] in_vector_293,
    input [7:0] in_vector_294,
    input [7:0] in_vector_295,
    input [7:0] in_vector_296,
    input [7:0] in_vector_297,
    input [7:0] in_vector_298,
    input [7:0] in_vector_299,
    input [7:0] in_vector_300,
    input [7:0] in_vector_301,
    input [7:0] in_vector_302,
    input [7:0] in_vector_303,
    input [7:0] in_vector_304,
    input [7:0] in_vector_305,
    input [7:0] in_vector_306,
    input [7:0] in_vector_307,
    input [7:0] in_vector_308,
    input [7:0] in_vector_309,
    input [7:0] in_vector_310,
    input [7:0] in_vector_311,
    input [7:0] in_vector_312,
    input [7:0] in_vector_313,
    input [7:0] in_vector_314,
    input [7:0] in_vector_315,
    input [7:0] in_vector_316,
    input [7:0] in_vector_317,
    input [7:0] in_vector_318,
    input [7:0] in_vector_319,
    input [7:0] in_vector_320,
    input [7:0] in_vector_321,
    input [7:0] in_vector_322,
    input [7:0] in_vector_323,
    input [7:0] in_vector_324,
    input [7:0] in_vector_325,
    input [7:0] in_vector_326,
    input [7:0] in_vector_327,
    input [7:0] in_vector_328,
    input [7:0] in_vector_329,
    input [7:0] in_vector_330,
    input [7:0] in_vector_331,
    input [7:0] in_vector_332,
    input [7:0] in_vector_333,
    input [7:0] in_vector_334,
    input [7:0] in_vector_335,
    input [7:0] in_vector_336,
    input [7:0] in_vector_337,
    input [7:0] in_vector_338,
    input [7:0] in_vector_339,
    input [7:0] in_vector_340,
    input [7:0] in_vector_341,
    input [7:0] in_vector_342,
    input [7:0] in_vector_343,
    input [7:0] in_vector_344,
    input [7:0] in_vector_345,
    input [7:0] in_vector_346,
    input [7:0] in_vector_347,
    input [7:0] in_vector_348,
    input [7:0] in_vector_349,
    input [7:0] in_vector_350,
    input [7:0] in_vector_351,
    input [7:0] in_vector_352,
    input [7:0] in_vector_353,
    input [7:0] in_vector_354,
    input [7:0] in_vector_355,
    input [7:0] in_vector_356,
    input [7:0] in_vector_357,
    input [7:0] in_vector_358,
    input [7:0] in_vector_359,
    input [7:0] in_vector_360,
    input [7:0] in_vector_361,
    input [7:0] in_vector_362,
    input [7:0] in_vector_363,
    input [7:0] in_vector_364,
    input [7:0] in_vector_365,
    input [7:0] in_vector_366,
    input [7:0] in_vector_367,
    input [7:0] in_vector_368,
    input [7:0] in_vector_369,
    input [7:0] in_vector_370,
    input [7:0] in_vector_371,
    input [7:0] in_vector_372,
    input [7:0] in_vector_373,
    input [7:0] in_vector_374,
    input [7:0] in_vector_375,
    input [7:0] in_vector_376,
    input [7:0] in_vector_377,
    input [7:0] in_vector_378,
    input [7:0] in_vector_379,
    input [7:0] in_vector_380,
    input [7:0] in_vector_381,
    input [7:0] in_vector_382,
    input [7:0] in_vector_383,
    input [7:0] in_vector_384,
    input [7:0] in_vector_385,
    input [7:0] in_vector_386,
    input [7:0] in_vector_387,
    input [7:0] in_vector_388,
    input [7:0] in_vector_389,
    input [7:0] in_vector_390,
    input [7:0] in_vector_391,
    input [7:0] in_vector_392,
    input [7:0] in_vector_393,
    input [7:0] in_vector_394,
    input [7:0] in_vector_395,
    input [7:0] in_vector_396,
    input [7:0] in_vector_397,
    input [7:0] in_vector_398,
    input [7:0] in_vector_399,
    input [7:0] in_vector_400,
    input [7:0] in_vector_401,
    input [7:0] in_vector_402,
    input [7:0] in_vector_403,
    input [7:0] in_vector_404,
    input [7:0] in_vector_405,
    input [7:0] in_vector_406,
    input [7:0] in_vector_407,
    input [7:0] in_vector_408,
    input [7:0] in_vector_409,
    input [7:0] in_vector_410,
    input [7:0] in_vector_411,
    input [7:0] in_vector_412,
    input [7:0] in_vector_413,
    input [7:0] in_vector_414,
    input [7:0] in_vector_415,
    input [7:0] in_vector_416,
    input [7:0] in_vector_417,
    input [7:0] in_vector_418,
    input [7:0] in_vector_419,
    input [7:0] in_vector_420,
    input [7:0] in_vector_421,
    input [7:0] in_vector_422,
    input [7:0] in_vector_423,
    input [7:0] in_vector_424,
    input [7:0] in_vector_425,
    input [7:0] in_vector_426,
    input [7:0] in_vector_427,
    input [7:0] in_vector_428,
    input [7:0] in_vector_429,
    input [7:0] in_vector_430,
    input [7:0] in_vector_431,
    input [7:0] in_vector_432,
    input [7:0] in_vector_433,
    input [7:0] in_vector_434,
    input [7:0] in_vector_435,
    input [7:0] in_vector_436,
    input [7:0] in_vector_437,
    input [7:0] in_vector_438,
    input [7:0] in_vector_439,
    input [7:0] in_vector_440,
    input [7:0] in_vector_441,
    input [7:0] in_vector_442,
    input [7:0] in_vector_443,
    input [7:0] in_vector_444,
    input [7:0] in_vector_445,
    input [7:0] in_vector_446,
    input [7:0] in_vector_447,
    input [7:0] in_vector_448,
    input [7:0] in_vector_449,
    input [7:0] in_vector_450,
    input [7:0] in_vector_451,
    input [7:0] in_vector_452,
    input [7:0] in_vector_453,
    input [7:0] in_vector_454,
    input [7:0] in_vector_455,
    input [7:0] in_vector_456,
    input [7:0] in_vector_457,
    input [7:0] in_vector_458,
    input [7:0] in_vector_459,
    input [7:0] in_vector_460,
    input [7:0] in_vector_461,
    input [7:0] in_vector_462,
    input [7:0] in_vector_463,
    input [7:0] in_vector_464,
    input [7:0] in_vector_465,
    input [7:0] in_vector_466,
    input [7:0] in_vector_467,
    input [7:0] in_vector_468,
    input [7:0] in_vector_469,
    input [7:0] in_vector_470,
    input [7:0] in_vector_471,
    input [7:0] in_vector_472,
    input [7:0] in_vector_473,
    input [7:0] in_vector_474,
    input [7:0] in_vector_475,
    input [7:0] in_vector_476,
    input [7:0] in_vector_477,
    input [7:0] in_vector_478,
    input [7:0] in_vector_479,
    input [7:0] in_vector_480,
    input [7:0] in_vector_481,
    input [7:0] in_vector_482,
    input [7:0] in_vector_483,
    input [7:0] in_vector_484,
    input [7:0] in_vector_485,
    input [7:0] in_vector_486,
    input [7:0] in_vector_487,
    input [7:0] in_vector_488,
    input [7:0] in_vector_489,
    input [7:0] in_vector_490,
    input [7:0] in_vector_491,
    input [7:0] in_vector_492,
    input [7:0] in_vector_493,
    input [7:0] in_vector_494,
    input [7:0] in_vector_495,
    input [7:0] in_vector_496,
    input [7:0] in_vector_497,
    input [7:0] in_vector_498,
    input [7:0] in_vector_499,
    input [7:0] in_vector_500,
    input [7:0] in_vector_501,
    input [7:0] in_vector_502,
    input [7:0] in_vector_503,
    input [7:0] in_vector_504,
    input [7:0] in_vector_505,
    input [7:0] in_vector_506,
    input [7:0] in_vector_507,
    input [7:0] in_vector_508,
    input [7:0] in_vector_509,
    input [7:0] in_vector_510,
    input [7:0] in_vector_511,
    input [7:0] in_vector_512,
    input [7:0] in_vector_513,
    input [7:0] in_vector_514,
    input [7:0] in_vector_515,
    input [7:0] in_vector_516,
    input [7:0] in_vector_517,
    input [7:0] in_vector_518,
    input [7:0] in_vector_519,
    input [7:0] in_vector_520,
    input [7:0] in_vector_521,
    input [7:0] in_vector_522,
    input [7:0] in_vector_523,
    input [7:0] in_vector_524,
    input [7:0] in_vector_525,
    input [7:0] in_vector_526,
    input [7:0] in_vector_527,
    input [7:0] in_vector_528,
    input [7:0] in_vector_529,
    input [7:0] in_vector_530,
    input [7:0] in_vector_531,
    input [7:0] in_vector_532,
    input [7:0] in_vector_533,
    input [7:0] in_vector_534,
    input [7:0] in_vector_535,
    input [7:0] in_vector_536,
    input [7:0] in_vector_537,
    input [7:0] in_vector_538,
    input [7:0] in_vector_539,
    input [7:0] in_vector_540,
    input [7:0] in_vector_541,
    input [7:0] in_vector_542,
    input [7:0] in_vector_543,
    input [7:0] in_vector_544,
    input [7:0] in_vector_545,
    input [7:0] in_vector_546,
    input [7:0] in_vector_547,
    input [7:0] in_vector_548,
    input [7:0] in_vector_549,
    input [7:0] in_vector_550,
    input [7:0] in_vector_551,
    input [7:0] in_vector_552,
    input [7:0] in_vector_553,
    input [7:0] in_vector_554,
    input [7:0] in_vector_555,
    input [7:0] in_vector_556,
    input [7:0] in_vector_557,
    input [7:0] in_vector_558,
    input [7:0] in_vector_559,
    input [7:0] in_vector_560,
    input [7:0] in_vector_561,
    input [7:0] in_vector_562,
    input [7:0] in_vector_563,
    input [7:0] in_vector_564,
    input [7:0] in_vector_565,
    input [7:0] in_vector_566,
    input [7:0] in_vector_567,
    input [7:0] in_vector_568,
    input [7:0] in_vector_569,
    input [7:0] in_vector_570,
    input [7:0] in_vector_571,
    input [7:0] in_vector_572,
    input [7:0] in_vector_573,
    input [7:0] in_vector_574,
    input [7:0] in_vector_575,
    input [7:0] in_vector_576,
    input [7:0] in_vector_577,
    input [7:0] in_vector_578,
    input [7:0] in_vector_579,
    input [7:0] in_vector_580,
    input [7:0] in_vector_581,
    input [7:0] in_vector_582,
    input [7:0] in_vector_583,
    input [7:0] in_vector_584,
    input [7:0] in_vector_585,
    input [7:0] in_vector_586,
    input [7:0] in_vector_587,
    input [7:0] in_vector_588,
    input [7:0] in_vector_589,
    input [7:0] in_vector_590,
    input [7:0] in_vector_591,
    input [7:0] in_vector_592,
    input [7:0] in_vector_593,
    input [7:0] in_vector_594,
    input [7:0] in_vector_595,
    input [7:0] in_vector_596,
    input [7:0] in_vector_597,
    input [7:0] in_vector_598,
    input [7:0] in_vector_599,
    input [7:0] in_vector_600,
    input [7:0] in_vector_601,
    input [7:0] in_vector_602,
    input [7:0] in_vector_603,
    input [7:0] in_vector_604,
    input [7:0] in_vector_605,
    input [7:0] in_vector_606,
    input [7:0] in_vector_607,
    input [7:0] in_vector_608,
    input [7:0] in_vector_609,
    input [7:0] in_vector_610,
    input [7:0] in_vector_611,
    input [7:0] in_vector_612,
    input [7:0] in_vector_613,
    input [7:0] in_vector_614,
    input [7:0] in_vector_615,
    input [7:0] in_vector_616,
    input [7:0] in_vector_617,
    input [7:0] in_vector_618,
    input [7:0] in_vector_619,
    input [7:0] in_vector_620,
    input [7:0] in_vector_621,
    input [7:0] in_vector_622,
    input [7:0] in_vector_623,
    input [7:0] in_vector_624,
    input [7:0] in_vector_625,
    input [7:0] in_vector_626,
    input [7:0] in_vector_627,
    input [7:0] in_vector_628,
    input [7:0] in_vector_629,
    input [7:0] in_vector_630,
    input [7:0] in_vector_631,
    input [7:0] in_vector_632,
    input [7:0] in_vector_633,
    input [7:0] in_vector_634,
    input [7:0] in_vector_635,
    input [7:0] in_vector_636,
    input [7:0] in_vector_637,
    input [7:0] in_vector_638,
    input [7:0] in_vector_639,
    input [7:0] in_vector_640,
    input [7:0] in_vector_641,
    input [7:0] in_vector_642,
    input [7:0] in_vector_643,
    input [7:0] in_vector_644,
    input [7:0] in_vector_645,
    input [7:0] in_vector_646,
    input [7:0] in_vector_647,
    input [7:0] in_vector_648,
    input [7:0] in_vector_649,
    input [7:0] in_vector_650,
    input [7:0] in_vector_651,
    input [7:0] in_vector_652,
    input [7:0] in_vector_653,
    input [7:0] in_vector_654,
    input [7:0] in_vector_655,
    input [7:0] in_vector_656,
    input [7:0] in_vector_657,
    input [7:0] in_vector_658,
    input [7:0] in_vector_659,
    input [7:0] in_vector_660,
    input [7:0] in_vector_661,
    input [7:0] in_vector_662,
    input [7:0] in_vector_663,
    input [7:0] in_vector_664,
    input [7:0] in_vector_665,
    input [7:0] in_vector_666,
    input [7:0] in_vector_667,
    input [7:0] in_vector_668,
    input [7:0] in_vector_669,
    input [7:0] in_vector_670,
    input [7:0] in_vector_671,
    input [7:0] in_vector_672,
    input [7:0] in_vector_673,
    input [7:0] in_vector_674,
    input [7:0] in_vector_675,
    input [7:0] in_vector_676,
    input [7:0] in_vector_677,
    input [7:0] in_vector_678,
    input [7:0] in_vector_679,
    input [7:0] in_vector_680,
    input [7:0] in_vector_681,
    input [7:0] in_vector_682,
    input [7:0] in_vector_683,
    input [7:0] in_vector_684,
    input [7:0] in_vector_685,
    input [7:0] in_vector_686,
    input [7:0] in_vector_687,
    input [7:0] in_vector_688,
    input [7:0] in_vector_689,
    input [7:0] in_vector_690,
    input [7:0] in_vector_691,
    input [7:0] in_vector_692,
    input [7:0] in_vector_693,
    input [7:0] in_vector_694,
    input [7:0] in_vector_695,
    input [7:0] in_vector_696,
    input [7:0] in_vector_697,
    input [7:0] in_vector_698,
    input [7:0] in_vector_699,
    input [7:0] in_vector_700,
    input [7:0] in_vector_701,
    input [7:0] in_vector_702,
    input [7:0] in_vector_703,
    input [7:0] in_vector_704,
    input [7:0] in_vector_705,
    input [7:0] in_vector_706,
    input [7:0] in_vector_707,
    input [7:0] in_vector_708,
    input [7:0] in_vector_709,
    input [7:0] in_vector_710,
    input [7:0] in_vector_711,
    input [7:0] in_vector_712,
    input [7:0] in_vector_713,
    input [7:0] in_vector_714,
    input [7:0] in_vector_715,
    input [7:0] in_vector_716,
    input [7:0] in_vector_717,
    input [7:0] in_vector_718,
    input [7:0] in_vector_719,
    input [7:0] in_vector_720,
    input [7:0] in_vector_721,
    input [7:0] in_vector_722,
    input [7:0] in_vector_723,
    input [7:0] in_vector_724,
    input [7:0] in_vector_725,
    input [7:0] in_vector_726,
    input [7:0] in_vector_727,
    input [7:0] in_vector_728,
    input [7:0] in_vector_729,
    input [7:0] in_vector_730,
    input [7:0] in_vector_731,
    input [7:0] in_vector_732,
    input [7:0] in_vector_733,
    input [7:0] in_vector_734,
    input [7:0] in_vector_735,
    input [7:0] in_vector_736,
    input [7:0] in_vector_737,
    input [7:0] in_vector_738,
    input [7:0] in_vector_739,
    input [7:0] in_vector_740,
    input [7:0] in_vector_741,
    input [7:0] in_vector_742,
    input [7:0] in_vector_743,
    input [7:0] in_vector_744,
    input [7:0] in_vector_745,
    input [7:0] in_vector_746,
    input [7:0] in_vector_747,
    input [7:0] in_vector_748,
    input [7:0] in_vector_749,
    input [7:0] in_vector_750,
    input [7:0] in_vector_751,
    input [7:0] in_vector_752,
    input [7:0] in_vector_753,
    input [7:0] in_vector_754,
    input [7:0] in_vector_755,
    input [7:0] in_vector_756,
    input [7:0] in_vector_757,
    input [7:0] in_vector_758,
    input [7:0] in_vector_759,
    input [7:0] in_vector_760,
    input [7:0] in_vector_761,
    input [7:0] in_vector_762,
    input [7:0] in_vector_763,
    input [7:0] in_vector_764,
    input [7:0] in_vector_765,
    input [7:0] in_vector_766,
    input [7:0] in_vector_767,
    input [7:0] in_vector_768,
    input [7:0] in_vector_769,
    input [7:0] in_vector_770,
    input [7:0] in_vector_771,
    input [7:0] in_vector_772,
    input [7:0] in_vector_773,
    input [7:0] in_vector_774,
    input [7:0] in_vector_775,
    input [7:0] in_vector_776,
    input [7:0] in_vector_777,
    input [7:0] in_vector_778,
    input [7:0] in_vector_779,
    input [7:0] in_vector_780,
    input [7:0] in_vector_781,
    input [7:0] in_vector_782,
    input [7:0] in_vector_783,

    // 3x3 filter weights
    input [7:0] weight_matrix_0,
    input [7:0] weight_matrix_1,
    input [7:0] weight_matrix_2,
    input [7:0] weight_matrix_3,
    input [7:0] weight_matrix_4,
    input [7:0] weight_matrix_5,
    input [7:0] weight_matrix_6,
    input [7:0] weight_matrix_7,
    input [7:0] weight_matrix_8,

    output reg [15:0] output_vector
);

    // Sliding window position (top-left of 3x3) over a 28x28 image -> 26x26 outputs
    reg [4:0] win_row;  // 0..25
    reg [4:0] win_col;  // 0..25

    // Helper: get pixel by flat index 0..783
    function [7:0] pixel;
        input integer idx;
        begin
            case (idx)
                0: pixel = in_vector_0;
                1: pixel = in_vector_1;
                2: pixel = in_vector_2;
                3: pixel = in_vector_3;
                4: pixel = in_vector_4;
                5: pixel = in_vector_5;
                6: pixel = in_vector_6;
                7: pixel = in_vector_7;
                8: pixel = in_vector_8;
                9: pixel = in_vector_9;
                10: pixel = in_vector_10;
                11: pixel = in_vector_11;
                12: pixel = in_vector_12;
                13: pixel = in_vector_13;
                14: pixel = in_vector_14;
                15: pixel = in_vector_15;
                16: pixel = in_vector_16;
                17: pixel = in_vector_17;
                18: pixel = in_vector_18;
                19: pixel = in_vector_19;
                20: pixel = in_vector_20;
                21: pixel = in_vector_21;
                22: pixel = in_vector_22;
                23: pixel = in_vector_23;
                24: pixel = in_vector_24;
                25: pixel = in_vector_25;
                26: pixel = in_vector_26;
                27: pixel = in_vector_27;
                28: pixel = in_vector_28;
                29: pixel = in_vector_29;
                30: pixel = in_vector_30;
                31: pixel = in_vector_31;
                32: pixel = in_vector_32;
                33: pixel = in_vector_33;
                34: pixel = in_vector_34;
                35: pixel = in_vector_35;
                36: pixel = in_vector_36;
                37: pixel = in_vector_37;
                38: pixel = in_vector_38;
                39: pixel = in_vector_39;
                40: pixel = in_vector_40;
                41: pixel = in_vector_41;
                42: pixel = in_vector_42;
                43: pixel = in_vector_43;
                44: pixel = in_vector_44;
                45: pixel = in_vector_45;
                46: pixel = in_vector_46;
                47: pixel = in_vector_47;
                48: pixel = in_vector_48;
                49: pixel = in_vector_49;
                50: pixel = in_vector_50;
                51: pixel = in_vector_51;
                52: pixel = in_vector_52;
                53: pixel = in_vector_53;
                54: pixel = in_vector_54;
                55: pixel = in_vector_55;
                56: pixel = in_vector_56;
                57: pixel = in_vector_57;
                58: pixel = in_vector_58;
                59: pixel = in_vector_59;
                60: pixel = in_vector_60;
                61: pixel = in_vector_61;
                62: pixel = in_vector_62;
                63: pixel = in_vector_63;
                64: pixel = in_vector_64;
                65: pixel = in_vector_65;
                66: pixel = in_vector_66;
                67: pixel = in_vector_67;
                68: pixel = in_vector_68;
                69: pixel = in_vector_69;
                70: pixel = in_vector_70;
                71: pixel = in_vector_71;
                72: pixel = in_vector_72;
                73: pixel = in_vector_73;
                74: pixel = in_vector_74;
                75: pixel = in_vector_75;
                76: pixel = in_vector_76;
                77: pixel = in_vector_77;
                78: pixel = in_vector_78;
                79: pixel = in_vector_79;
                80: pixel = in_vector_80;
                81: pixel = in_vector_81;
                82: pixel = in_vector_82;
                83: pixel = in_vector_83;
                84: pixel = in_vector_84;
                85: pixel = in_vector_85;
                86: pixel = in_vector_86;
                87: pixel = in_vector_87;
                88: pixel = in_vector_88;
                89: pixel = in_vector_89;
                90: pixel = in_vector_90;
                91: pixel = in_vector_91;
                92: pixel = in_vector_92;
                93: pixel = in_vector_93;
                94: pixel = in_vector_94;
                95: pixel = in_vector_95;
                96: pixel = in_vector_96;
                97: pixel = in_vector_97;
                98: pixel = in_vector_98;
                99: pixel = in_vector_99;
                100: pixel = in_vector_100;
                101: pixel = in_vector_101;
                102: pixel = in_vector_102;
                103: pixel = in_vector_103;
                104: pixel = in_vector_104;
                105: pixel = in_vector_105;
                106: pixel = in_vector_106;
                107: pixel = in_vector_107;
                108: pixel = in_vector_108;
                109: pixel = in_vector_109;
                110: pixel = in_vector_110;
                111: pixel = in_vector_111;
                112: pixel = in_vector_112;
                113: pixel = in_vector_113;
                114: pixel = in_vector_114;
                115: pixel = in_vector_115;
                116: pixel = in_vector_116;
                117: pixel = in_vector_117;
                118: pixel = in_vector_118;
                119: pixel = in_vector_119;
                120: pixel = in_vector_120;
                121: pixel = in_vector_121;
                122: pixel = in_vector_122;
                123: pixel = in_vector_123;
                124: pixel = in_vector_124;
                125: pixel = in_vector_125;
                126: pixel = in_vector_126;
                127: pixel = in_vector_127;
                128: pixel = in_vector_128;
                129: pixel = in_vector_129;
                130: pixel = in_vector_130;
                131: pixel = in_vector_131;
                132: pixel = in_vector_132;
                133: pixel = in_vector_133;
                134: pixel = in_vector_134;
                135: pixel = in_vector_135;
                136: pixel = in_vector_136;
                137: pixel = in_vector_137;
                138: pixel = in_vector_138;
                139: pixel = in_vector_139;
                140: pixel = in_vector_140;
                141: pixel = in_vector_141;
                142: pixel = in_vector_142;
                143: pixel = in_vector_143;
                144: pixel = in_vector_144;
                145: pixel = in_vector_145;
                146: pixel = in_vector_146;
                147: pixel = in_vector_147;
                148: pixel = in_vector_148;
                149: pixel = in_vector_149;
                150: pixel = in_vector_150;
                151: pixel = in_vector_151;
                152: pixel = in_vector_152;
                153: pixel = in_vector_153;
                154: pixel = in_vector_154;
                155: pixel = in_vector_155;
                156: pixel = in_vector_156;
                157: pixel = in_vector_157;
                158: pixel = in_vector_158;
                159: pixel = in_vector_159;
                160: pixel = in_vector_160;
                161: pixel = in_vector_161;
                162: pixel = in_vector_162;
                163: pixel = in_vector_163;
                164: pixel = in_vector_164;
                165: pixel = in_vector_165;
                166: pixel = in_vector_166;
                167: pixel = in_vector_167;
                168: pixel = in_vector_168;
                169: pixel = in_vector_169;
                170: pixel = in_vector_170;
                171: pixel = in_vector_171;
                172: pixel = in_vector_172;
                173: pixel = in_vector_173;
                174: pixel = in_vector_174;
                175: pixel = in_vector_175;
                176: pixel = in_vector_176;
                177: pixel = in_vector_177;
                178: pixel = in_vector_178;
                179: pixel = in_vector_179;
                180: pixel = in_vector_180;
                181: pixel = in_vector_181;
                182: pixel = in_vector_182;
                183: pixel = in_vector_183;
                184: pixel = in_vector_184;
                185: pixel = in_vector_185;
                186: pixel = in_vector_186;
                187: pixel = in_vector_187;
                188: pixel = in_vector_188;
                189: pixel = in_vector_189;
                190: pixel = in_vector_190;
                191: pixel = in_vector_191;
                192: pixel = in_vector_192;
                193: pixel = in_vector_193;
                194: pixel = in_vector_194;
                195: pixel = in_vector_195;
                196: pixel = in_vector_196;
                197: pixel = in_vector_197;
                198: pixel = in_vector_198;
                199: pixel = in_vector_199;
                200: pixel = in_vector_200;
                201: pixel = in_vector_201;
                202: pixel = in_vector_202;
                203: pixel = in_vector_203;
                204: pixel = in_vector_204;
                205: pixel = in_vector_205;
                206: pixel = in_vector_206;
                207: pixel = in_vector_207;
                208: pixel = in_vector_208;
                209: pixel = in_vector_209;
                210: pixel = in_vector_210;
                211: pixel = in_vector_211;
                212: pixel = in_vector_212;
                213: pixel = in_vector_213;
                214: pixel = in_vector_214;
                215: pixel = in_vector_215;
                216: pixel = in_vector_216;
                217: pixel = in_vector_217;
                218: pixel = in_vector_218;
                219: pixel = in_vector_219;
                220: pixel = in_vector_220;
                221: pixel = in_vector_221;
                222: pixel = in_vector_222;
                223: pixel = in_vector_223;
                224: pixel = in_vector_224;
                225: pixel = in_vector_225;
                226: pixel = in_vector_226;
                227: pixel = in_vector_227;
                228: pixel = in_vector_228;
                229: pixel = in_vector_229;
                230: pixel = in_vector_230;
                231: pixel = in_vector_231;
                232: pixel = in_vector_232;
                233: pixel = in_vector_233;
                234: pixel = in_vector_234;
                235: pixel = in_vector_235;
                236: pixel = in_vector_236;
                237: pixel = in_vector_237;
                238: pixel = in_vector_238;
                239: pixel = in_vector_239;
                240: pixel = in_vector_240;
                241: pixel = in_vector_241;
                242: pixel = in_vector_242;
                243: pixel = in_vector_243;
                244: pixel = in_vector_244;
                245: pixel = in_vector_245;
                246: pixel = in_vector_246;
                247: pixel = in_vector_247;
                248: pixel = in_vector_248;
                249: pixel = in_vector_249;
                250: pixel = in_vector_250;
                251: pixel = in_vector_251;
                252: pixel = in_vector_252;
                253: pixel = in_vector_253;
                254: pixel = in_vector_254;
                255: pixel = in_vector_255;
                256: pixel = in_vector_256;
                257: pixel = in_vector_257;
                258: pixel = in_vector_258;
                259: pixel = in_vector_259;
                260: pixel = in_vector_260;
                261: pixel = in_vector_261;
                262: pixel = in_vector_262;
                263: pixel = in_vector_263;
                264: pixel = in_vector_264;
                265: pixel = in_vector_265;
                266: pixel = in_vector_266;
                267: pixel = in_vector_267;
                268: pixel = in_vector_268;
                269: pixel = in_vector_269;
                270: pixel = in_vector_270;
                271: pixel = in_vector_271;
                272: pixel = in_vector_272;
                273: pixel = in_vector_273;
                274: pixel = in_vector_274;
                275: pixel = in_vector_275;
                276: pixel = in_vector_276;
                277: pixel = in_vector_277;
                278: pixel = in_vector_278;
                279: pixel = in_vector_279;
                280: pixel = in_vector_280;
                281: pixel = in_vector_281;
                282: pixel = in_vector_282;
                283: pixel = in_vector_283;
                284: pixel = in_vector_284;
                285: pixel = in_vector_285;
                286: pixel = in_vector_286;
                287: pixel = in_vector_287;
                288: pixel = in_vector_288;
                289: pixel = in_vector_289;
                290: pixel = in_vector_290;
                291: pixel = in_vector_291;
                292: pixel = in_vector_292;
                293: pixel = in_vector_293;
                294: pixel = in_vector_294;
                295: pixel = in_vector_295;
                296: pixel = in_vector_296;
                297: pixel = in_vector_297;
                298: pixel = in_vector_298;
                299: pixel = in_vector_299;
                300: pixel = in_vector_300;
                301: pixel = in_vector_301;
                302: pixel = in_vector_302;
                303: pixel = in_vector_303;
                304: pixel = in_vector_304;
                305: pixel = in_vector_305;
                306: pixel = in_vector_306;
                307: pixel = in_vector_307;
                308: pixel = in_vector_308;
                309: pixel = in_vector_309;
                310: pixel = in_vector_310;
                311: pixel = in_vector_311;
                312: pixel = in_vector_312;
                313: pixel = in_vector_313;
                314: pixel = in_vector_314;
                315: pixel = in_vector_315;
                316: pixel = in_vector_316;
                317: pixel = in_vector_317;
                318: pixel = in_vector_318;
                319: pixel = in_vector_319;
                320: pixel = in_vector_320;
                321: pixel = in_vector_321;
                322: pixel = in_vector_322;
                323: pixel = in_vector_323;
                324: pixel = in_vector_324;
                325: pixel = in_vector_325;
                326: pixel = in_vector_326;
                327: pixel = in_vector_327;
                328: pixel = in_vector_328;
                329: pixel = in_vector_329;
                330: pixel = in_vector_330;
                331: pixel = in_vector_331;
                332: pixel = in_vector_332;
                333: pixel = in_vector_333;
                334: pixel = in_vector_334;
                335: pixel = in_vector_335;
                336: pixel = in_vector_336;
                337: pixel = in_vector_337;
                338: pixel = in_vector_338;
                339: pixel = in_vector_339;
                340: pixel = in_vector_340;
                341: pixel = in_vector_341;
                342: pixel = in_vector_342;
                343: pixel = in_vector_343;
                344: pixel = in_vector_344;
                345: pixel = in_vector_345;
                346: pixel = in_vector_346;
                347: pixel = in_vector_347;
                348: pixel = in_vector_348;
                349: pixel = in_vector_349;
                350: pixel = in_vector_350;
                351: pixel = in_vector_351;
                352: pixel = in_vector_352;
                353: pixel = in_vector_353;
                354: pixel = in_vector_354;
                355: pixel = in_vector_355;
                356: pixel = in_vector_356;
                357: pixel = in_vector_357;
                358: pixel = in_vector_358;
                359: pixel = in_vector_359;
                360: pixel = in_vector_360;
                361: pixel = in_vector_361;
                362: pixel = in_vector_362;
                363: pixel = in_vector_363;
                364: pixel = in_vector_364;
                365: pixel = in_vector_365;
                366: pixel = in_vector_366;
                367: pixel = in_vector_367;
                368: pixel = in_vector_368;
                369: pixel = in_vector_369;
                370: pixel = in_vector_370;
                371: pixel = in_vector_371;
                372: pixel = in_vector_372;
                373: pixel = in_vector_373;
                374: pixel = in_vector_374;
                375: pixel = in_vector_375;
                376: pixel = in_vector_376;
                377: pixel = in_vector_377;
                378: pixel = in_vector_378;
                379: pixel = in_vector_379;
                380: pixel = in_vector_380;
                381: pixel = in_vector_381;
                382: pixel = in_vector_382;
                383: pixel = in_vector_383;
                384: pixel = in_vector_384;
                385: pixel = in_vector_385;
                386: pixel = in_vector_386;
                387: pixel = in_vector_387;
                388: pixel = in_vector_388;
                389: pixel = in_vector_389;
                390: pixel = in_vector_390;
                391: pixel = in_vector_391;
                392: pixel = in_vector_392;
                393: pixel = in_vector_393;
                394: pixel = in_vector_394;
                395: pixel = in_vector_395;
                396: pixel = in_vector_396;
                397: pixel = in_vector_397;
                398: pixel = in_vector_398;
                399: pixel = in_vector_399;
                400: pixel = in_vector_400;
                401: pixel = in_vector_401;
                402: pixel = in_vector_402;
                403: pixel = in_vector_403;
                404: pixel = in_vector_404;
                405: pixel = in_vector_405;
                406: pixel = in_vector_406;
                407: pixel = in_vector_407;
                408: pixel = in_vector_408;
                409: pixel = in_vector_409;
                410: pixel = in_vector_410;
                411: pixel = in_vector_411;
                412: pixel = in_vector_412;
                413: pixel = in_vector_413;
                414: pixel = in_vector_414;
                415: pixel = in_vector_415;
                416: pixel = in_vector_416;
                417: pixel = in_vector_417;
                418: pixel = in_vector_418;
                419: pixel = in_vector_419;
                420: pixel = in_vector_420;
                421: pixel = in_vector_421;
                422: pixel = in_vector_422;
                423: pixel = in_vector_423;
                424: pixel = in_vector_424;
                425: pixel = in_vector_425;
                426: pixel = in_vector_426;
                427: pixel = in_vector_427;
                428: pixel = in_vector_428;
                429: pixel = in_vector_429;
                430: pixel = in_vector_430;
                431: pixel = in_vector_431;
                432: pixel = in_vector_432;
                433: pixel = in_vector_433;
                434: pixel = in_vector_434;
                435: pixel = in_vector_435;
                436: pixel = in_vector_436;
                437: pixel = in_vector_437;
                438: pixel = in_vector_438;
                439: pixel = in_vector_439;
                440: pixel = in_vector_440;
                441: pixel = in_vector_441;
                442: pixel = in_vector_442;
                443: pixel = in_vector_443;
                444: pixel = in_vector_444;
                445: pixel = in_vector_445;
                446: pixel = in_vector_446;
                447: pixel = in_vector_447;
                448: pixel = in_vector_448;
                449: pixel = in_vector_449;
                450: pixel = in_vector_450;
                451: pixel = in_vector_451;
                452: pixel = in_vector_452;
                453: pixel = in_vector_453;
                454: pixel = in_vector_454;
                455: pixel = in_vector_455;
                456: pixel = in_vector_456;
                457: pixel = in_vector_457;
                458: pixel = in_vector_458;
                459: pixel = in_vector_459;
                460: pixel = in_vector_460;
                461: pixel = in_vector_461;
                462: pixel = in_vector_462;
                463: pixel = in_vector_463;
                464: pixel = in_vector_464;
                465: pixel = in_vector_465;
                466: pixel = in_vector_466;
                467: pixel = in_vector_467;
                468: pixel = in_vector_468;
                469: pixel = in_vector_469;
                470: pixel = in_vector_470;
                471: pixel = in_vector_471;
                472: pixel = in_vector_472;
                473: pixel = in_vector_473;
                474: pixel = in_vector_474;
                475: pixel = in_vector_475;
                476: pixel = in_vector_476;
                477: pixel = in_vector_477;
                478: pixel = in_vector_478;
                479: pixel = in_vector_479;
                480: pixel = in_vector_480;
                481: pixel = in_vector_481;
                482: pixel = in_vector_482;
                483: pixel = in_vector_483;
                484: pixel = in_vector_484;
                485: pixel = in_vector_485;
                486: pixel = in_vector_486;
                487: pixel = in_vector_487;
                488: pixel = in_vector_488;
                489: pixel = in_vector_489;
                490: pixel = in_vector_490;
                491: pixel = in_vector_491;
                492: pixel = in_vector_492;
                493: pixel = in_vector_493;
                494: pixel = in_vector_494;
                495: pixel = in_vector_495;
                496: pixel = in_vector_496;
                497: pixel = in_vector_497;
                498: pixel = in_vector_498;
                499: pixel = in_vector_499;
                500: pixel = in_vector_500;
                501: pixel = in_vector_501;
                502: pixel = in_vector_502;
                503: pixel = in_vector_503;
                504: pixel = in_vector_504;
                505: pixel = in_vector_505;
                506: pixel = in_vector_506;
                507: pixel = in_vector_507;
                508: pixel = in_vector_508;
                509: pixel = in_vector_509;
                510: pixel = in_vector_510;
                511: pixel = in_vector_511;
                512: pixel = in_vector_512;
                513: pixel = in_vector_513;
                514: pixel = in_vector_514;
                515: pixel = in_vector_515;
                516: pixel = in_vector_516;
                517: pixel = in_vector_517;
                518: pixel = in_vector_518;
                519: pixel = in_vector_519;
                520: pixel = in_vector_520;
                521: pixel = in_vector_521;
                522: pixel = in_vector_522;
                523: pixel = in_vector_523;
                524: pixel = in_vector_524;
                525: pixel = in_vector_525;
                526: pixel = in_vector_526;
                527: pixel = in_vector_527;
                528: pixel = in_vector_528;
                529: pixel = in_vector_529;
                530: pixel = in_vector_530;
                531: pixel = in_vector_531;
                532: pixel = in_vector_532;
                533: pixel = in_vector_533;
                534: pixel = in_vector_534;
                535: pixel = in_vector_535;
                536: pixel = in_vector_536;
                537: pixel = in_vector_537;
                538: pixel = in_vector_538;
                539: pixel = in_vector_539;
                540: pixel = in_vector_540;
                541: pixel = in_vector_541;
                542: pixel = in_vector_542;
                543: pixel = in_vector_543;
                544: pixel = in_vector_544;
                545: pixel = in_vector_545;
                546: pixel = in_vector_546;
                547: pixel = in_vector_547;
                548: pixel = in_vector_548;
                549: pixel = in_vector_549;
                550: pixel = in_vector_550;
                551: pixel = in_vector_551;
                552: pixel = in_vector_552;
                553: pixel = in_vector_553;
                554: pixel = in_vector_554;
                555: pixel = in_vector_555;
                556: pixel = in_vector_556;
                557: pixel = in_vector_557;
                558: pixel = in_vector_558;
                559: pixel = in_vector_559;
                560: pixel = in_vector_560;
                561: pixel = in_vector_561;
                562: pixel = in_vector_562;
                563: pixel = in_vector_563;
                564: pixel = in_vector_564;
                565: pixel = in_vector_565;
                566: pixel = in_vector_566;
                567: pixel = in_vector_567;
                568: pixel = in_vector_568;
                569: pixel = in_vector_569;
                570: pixel = in_vector_570;
                571: pixel = in_vector_571;
                572: pixel = in_vector_572;
                573: pixel = in_vector_573;
                574: pixel = in_vector_574;
                575: pixel = in_vector_575;
                576: pixel = in_vector_576;
                577: pixel = in_vector_577;
                578: pixel = in_vector_578;
                579: pixel = in_vector_579;
                580: pixel = in_vector_580;
                581: pixel = in_vector_581;
                582: pixel = in_vector_582;
                583: pixel = in_vector_583;
                584: pixel = in_vector_584;
                585: pixel = in_vector_585;
                586: pixel = in_vector_586;
                587: pixel = in_vector_587;
                588: pixel = in_vector_588;
                589: pixel = in_vector_589;
                590: pixel = in_vector_590;
                591: pixel = in_vector_591;
                592: pixel = in_vector_592;
                593: pixel = in_vector_593;
                594: pixel = in_vector_594;
                595: pixel = in_vector_595;
                596: pixel = in_vector_596;
                597: pixel = in_vector_597;
                598: pixel = in_vector_598;
                599: pixel = in_vector_599;
                600: pixel = in_vector_600;
                601: pixel = in_vector_601;
                602: pixel = in_vector_602;
                603: pixel = in_vector_603;
                604: pixel = in_vector_604;
                605: pixel = in_vector_605;
                606: pixel = in_vector_606;
                607: pixel = in_vector_607;
                608: pixel = in_vector_608;
                609: pixel = in_vector_609;
                610: pixel = in_vector_610;
                611: pixel = in_vector_611;
                612: pixel = in_vector_612;
                613: pixel = in_vector_613;
                614: pixel = in_vector_614;
                615: pixel = in_vector_615;
                616: pixel = in_vector_616;
                617: pixel = in_vector_617;
                618: pixel = in_vector_618;
                619: pixel = in_vector_619;
                620: pixel = in_vector_620;
                621: pixel = in_vector_621;
                622: pixel = in_vector_622;
                623: pixel = in_vector_623;
                624: pixel = in_vector_624;
                625: pixel = in_vector_625;
                626: pixel = in_vector_626;
                627: pixel = in_vector_627;
                628: pixel = in_vector_628;
                629: pixel = in_vector_629;
                630: pixel = in_vector_630;
                631: pixel = in_vector_631;
                632: pixel = in_vector_632;
                633: pixel = in_vector_633;
                634: pixel = in_vector_634;
                635: pixel = in_vector_635;
                636: pixel = in_vector_636;
                637: pixel = in_vector_637;
                638: pixel = in_vector_638;
                639: pixel = in_vector_639;
                640: pixel = in_vector_640;
                641: pixel = in_vector_641;
                642: pixel = in_vector_642;
                643: pixel = in_vector_643;
                644: pixel = in_vector_644;
                645: pixel = in_vector_645;
                646: pixel = in_vector_646;
                647: pixel = in_vector_647;
                648: pixel = in_vector_648;
                649: pixel = in_vector_649;
                650: pixel = in_vector_650;
                651: pixel = in_vector_651;
                652: pixel = in_vector_652;
                653: pixel = in_vector_653;
                654: pixel = in_vector_654;
                655: pixel = in_vector_655;
                656: pixel = in_vector_656;
                657: pixel = in_vector_657;
                658: pixel = in_vector_658;
                659: pixel = in_vector_659;
                660: pixel = in_vector_660;
                661: pixel = in_vector_661;
                662: pixel = in_vector_662;
                663: pixel = in_vector_663;
                664: pixel = in_vector_664;
                665: pixel = in_vector_665;
                666: pixel = in_vector_666;
                667: pixel = in_vector_667;
                668: pixel = in_vector_668;
                669: pixel = in_vector_669;
                670: pixel = in_vector_670;
                671: pixel = in_vector_671;
                672: pixel = in_vector_672;
                673: pixel = in_vector_673;
                674: pixel = in_vector_674;
                675: pixel = in_vector_675;
                676: pixel = in_vector_676;
                677: pixel = in_vector_677;
                678: pixel = in_vector_678;
                679: pixel = in_vector_679;
                680: pixel = in_vector_680;
                681: pixel = in_vector_681;
                682: pixel = in_vector_682;
                683: pixel = in_vector_683;
                684: pixel = in_vector_684;
                685: pixel = in_vector_685;
                686: pixel = in_vector_686;
                687: pixel = in_vector_687;
                688: pixel = in_vector_688;
                689: pixel = in_vector_689;
                690: pixel = in_vector_690;
                691: pixel = in_vector_691;
                692: pixel = in_vector_692;
                693: pixel = in_vector_693;
                694: pixel = in_vector_694;
                695: pixel = in_vector_695;
                696: pixel = in_vector_696;
                697: pixel = in_vector_697;
                698: pixel = in_vector_698;
                699: pixel = in_vector_699;
                700: pixel = in_vector_700;
                701: pixel = in_vector_701;
                702: pixel = in_vector_702;
                703: pixel = in_vector_703;
                704: pixel = in_vector_704;
                705: pixel = in_vector_705;
                706: pixel = in_vector_706;
                707: pixel = in_vector_707;
                708: pixel = in_vector_708;
                709: pixel = in_vector_709;
                710: pixel = in_vector_710;
                711: pixel = in_vector_711;
                712: pixel = in_vector_712;
                713: pixel = in_vector_713;
                714: pixel = in_vector_714;
                715: pixel = in_vector_715;
                716: pixel = in_vector_716;
                717: pixel = in_vector_717;
                718: pixel = in_vector_718;
                719: pixel = in_vector_719;
                720: pixel = in_vector_720;
                721: pixel = in_vector_721;
                722: pixel = in_vector_722;
                723: pixel = in_vector_723;
                724: pixel = in_vector_724;
                725: pixel = in_vector_725;
                726: pixel = in_vector_726;
                727: pixel = in_vector_727;
                728: pixel = in_vector_728;
                729: pixel = in_vector_729;
                730: pixel = in_vector_730;
                731: pixel = in_vector_731;
                732: pixel = in_vector_732;
                733: pixel = in_vector_733;
                734: pixel = in_vector_734;
                735: pixel = in_vector_735;
                736: pixel = in_vector_736;
                737: pixel = in_vector_737;
                738: pixel = in_vector_738;
                739: pixel = in_vector_739;
                740: pixel = in_vector_740;
                741: pixel = in_vector_741;
                742: pixel = in_vector_742;
                743: pixel = in_vector_743;
                744: pixel = in_vector_744;
                745: pixel = in_vector_745;
                746: pixel = in_vector_746;
                747: pixel = in_vector_747;
                748: pixel = in_vector_748;
                749: pixel = in_vector_749;
                750: pixel = in_vector_750;
                751: pixel = in_vector_751;
                752: pixel = in_vector_752;
                753: pixel = in_vector_753;
                754: pixel = in_vector_754;
                755: pixel = in_vector_755;
                756: pixel = in_vector_756;
                757: pixel = in_vector_757;
                758: pixel = in_vector_758;
                759: pixel = in_vector_759;
                760: pixel = in_vector_760;
                761: pixel = in_vector_761;
                762: pixel = in_vector_762;
                763: pixel = in_vector_763;
                764: pixel = in_vector_764;
                765: pixel = in_vector_765;
                766: pixel = in_vector_766;
                767: pixel = in_vector_767;
                768: pixel = in_vector_768;
                769: pixel = in_vector_769;
                770: pixel = in_vector_770;
                771: pixel = in_vector_771;
                772: pixel = in_vector_772;
                773: pixel = in_vector_773;
                774: pixel = in_vector_774;
                775: pixel = in_vector_775;
                776: pixel = in_vector_776;
                777: pixel = in_vector_777;
                778: pixel = in_vector_778;
                779: pixel = in_vector_779;
                780: pixel = in_vector_780;
                781: pixel = in_vector_781;
                782: pixel = in_vector_782;
                783: pixel = in_vector_783;
                default: pixel = 8'd0;
            endcase
        end
    endfunction

    // Update sliding window counters once per clock
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            win_row <= 5'd0;
            win_col <= 5'd0;
        end else begin
            if (win_col == 5'd25) begin
                win_col <= 5'd0;
                if (win_row == 5'd25) win_row <= 5'd0;
                else win_row <= win_row + 5'd1;
            end else begin
                win_col <= win_col + 5'd1;
            end
        end
    end

    // Combinational 3x3 MAC for current window position
    integer base;
    reg [31:0] acc;
    always @(*) begin
        base = (win_row * 28) + win_col;
        acc  = 32'd0;

        // 3x3 window:
        // [base+0  base+1  base+2]
        // [base+28 base+29 base+30]
        // [base+56 base+57 base+58]
        acc = acc + (pixel(base + 0 ) * weight_matrix_0);
        acc = acc + (pixel(base + 1 ) * weight_matrix_1);
        acc = acc + (pixel(base + 2 ) * weight_matrix_2);

        acc = acc + (pixel(base + 28) * weight_matrix_3);
        acc = acc + (pixel(base + 29) * weight_matrix_4);
        acc = acc + (pixel(base + 30) * weight_matrix_5);

        acc = acc + (pixel(base + 56) * weight_matrix_6);
        acc = acc + (pixel(base + 57) * weight_matrix_7);
        acc = acc + (pixel(base + 58) * weight_matrix_8);

      
    end
	
	// Register the output so it updates cleanly on clock, and stays known during reset
    always @(posedge clk or posedge rst) begin
    if (rst) begin
        output_vector <= 16'd0;
    end else begin
        output_vector <= acc[15:0];
    end
end

endmodule
