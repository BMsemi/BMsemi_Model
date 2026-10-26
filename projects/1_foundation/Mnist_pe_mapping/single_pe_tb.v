module tb_PE;

    // Parameters
    parameter Y = 28;  // Image height (28x28)
    parameter X = 28;  // Image width (28x28)

    // Input Signals
    reg clk;
    reg rst;
    reg [7:0] in_vector [0:783];  // Flattened 28x28 input image (grayscale) - 784 elements
    reg [7:0] weight_matrix_0, weight_matrix_1, weight_matrix_2;
    reg [7:0] weight_matrix_3, weight_matrix_4, weight_matrix_5;
    reg [7:0] weight_matrix_6, weight_matrix_7, weight_matrix_8;  // 3x3 weights

    // Output Signals
    wire [15:0] pe_result;  // Output feature map for the single PE
        reg [15:0] output_vector [0:25][0:25]; // Output feature map (26x26)

    integer i, j, k, row, col;

    // Instantiate the single PE
    PE pe_inst (
        .clk(clk),
        .rst(rst),
        .in_vector_0(in_vector[0]),
        .in_vector_1(in_vector[1]),
        .in_vector_2(in_vector[2]),
        .in_vector_3(in_vector[3]),
        .in_vector_4(in_vector[4]),
        .in_vector_5(in_vector[5]),
        .in_vector_6(in_vector[6]),
        .in_vector_7(in_vector[7]),
        .in_vector_8(in_vector[8]),
        .in_vector_9(in_vector[9]),
        .in_vector_10(in_vector[10]),
        .in_vector_11(in_vector[11]),
        .in_vector_12(in_vector[12]),
        .in_vector_13(in_vector[13]),
        .in_vector_14(in_vector[14]),
        .in_vector_15(in_vector[15]),
        .in_vector_16(in_vector[16]),
        .in_vector_17(in_vector[17]),
        .in_vector_18(in_vector[18]),
        .in_vector_19(in_vector[19]),
        .in_vector_20(in_vector[20]),
        .in_vector_21(in_vector[21]),
        .in_vector_22(in_vector[22]),
        .in_vector_23(in_vector[23]),
        .in_vector_24(in_vector[24]),
        .in_vector_25(in_vector[25]),
        .in_vector_26(in_vector[26]),
        .in_vector_27(in_vector[27]),
        .in_vector_28(in_vector[28]),
        .in_vector_29(in_vector[29]),
        .in_vector_30(in_vector[30]),
        .in_vector_31(in_vector[31]),
        .in_vector_32(in_vector[32]),
        .in_vector_33(in_vector[33]),
        .in_vector_34(in_vector[34]),
        .in_vector_35(in_vector[35]),
        .in_vector_36(in_vector[36]),
        .in_vector_37(in_vector[37]),
        .in_vector_38(in_vector[38]),
        .in_vector_39(in_vector[39]),
        .in_vector_40(in_vector[40]),
        .in_vector_41(in_vector[41]),
        .in_vector_42(in_vector[42]),
        .in_vector_43(in_vector[43]),
        .in_vector_44(in_vector[44]),
        .in_vector_45(in_vector[45]),
        .in_vector_46(in_vector[46]),
        .in_vector_47(in_vector[47]),
        .in_vector_48(in_vector[48]),
        .in_vector_49(in_vector[49]),
        .in_vector_50(in_vector[50]),
        .in_vector_51(in_vector[51]),
        .in_vector_52(in_vector[52]),
        .in_vector_53(in_vector[53]),
        .in_vector_54(in_vector[54]),
        .in_vector_55(in_vector[55]),
        .in_vector_56(in_vector[56]),
        .in_vector_57(in_vector[57]),
        .in_vector_58(in_vector[58]),
        .in_vector_59(in_vector[59]),
        .in_vector_60(in_vector[60]),
        .in_vector_61(in_vector[61]),
        .in_vector_62(in_vector[62]),
        .in_vector_63(in_vector[63]),
        .in_vector_64(in_vector[64]),
        .in_vector_65(in_vector[65]),
        .in_vector_66(in_vector[66]),
        .in_vector_67(in_vector[67]),
        .in_vector_68(in_vector[68]),
        .in_vector_69(in_vector[69]),
        .in_vector_70(in_vector[70]),
        .in_vector_71(in_vector[71]),
        .in_vector_72(in_vector[72]),
        .in_vector_73(in_vector[73]),
        .in_vector_74(in_vector[74]),
        .in_vector_75(in_vector[75]),
        .in_vector_76(in_vector[76]),
        .in_vector_77(in_vector[77]),
        .in_vector_78(in_vector[78]),
        .in_vector_79(in_vector[79]),
        .in_vector_80(in_vector[80]),
        .in_vector_81(in_vector[81]),
        .in_vector_82(in_vector[82]),
        .in_vector_83(in_vector[83]),
        .in_vector_84(in_vector[84]),
        .in_vector_85(in_vector[85]),
        .in_vector_86(in_vector[86]),
        .in_vector_87(in_vector[87]),
        .in_vector_88(in_vector[88]),
        .in_vector_89(in_vector[89]),
        .in_vector_90(in_vector[90]),
        .in_vector_91(in_vector[91]),
        .in_vector_92(in_vector[92]),
        .in_vector_93(in_vector[93]),
        .in_vector_94(in_vector[94]),
        .in_vector_95(in_vector[95]),
        .in_vector_96(in_vector[96]),
        .in_vector_97(in_vector[97]),
        .in_vector_98(in_vector[98]),
        .in_vector_99(in_vector[99]),
        .in_vector_100(in_vector[100]),
        .in_vector_101(in_vector[101]),
        .in_vector_102(in_vector[102]),
        .in_vector_103(in_vector[103]),
        .in_vector_104(in_vector[104]),
        .in_vector_105(in_vector[105]),
        .in_vector_106(in_vector[106]),
        .in_vector_107(in_vector[107]),
        .in_vector_108(in_vector[108]),
        .in_vector_109(in_vector[109]),
        .in_vector_110(in_vector[110]),
        .in_vector_111(in_vector[111]),
        .in_vector_112(in_vector[112]),
        .in_vector_113(in_vector[113]),
        .in_vector_114(in_vector[114]),
        .in_vector_115(in_vector[115]),
        .in_vector_116(in_vector[116]),
        .in_vector_117(in_vector[117]),
        .in_vector_118(in_vector[118]),
        .in_vector_119(in_vector[119]),
        .in_vector_120(in_vector[120]),
        .in_vector_121(in_vector[121]),
        .in_vector_122(in_vector[122]),
        .in_vector_123(in_vector[123]),
        .in_vector_124(in_vector[124]),
        .in_vector_125(in_vector[125]),
        .in_vector_126(in_vector[126]),
        .in_vector_127(in_vector[127]),
        .in_vector_128(in_vector[128]),
        .in_vector_129(in_vector[129]),
        .in_vector_130(in_vector[130]),
        .in_vector_131(in_vector[131]),
        .in_vector_132(in_vector[132]),
        .in_vector_133(in_vector[133]),
        .in_vector_134(in_vector[134]),
        .in_vector_135(in_vector[135]),
        .in_vector_136(in_vector[136]),
        .in_vector_137(in_vector[137]),
        .in_vector_138(in_vector[138]),
        .in_vector_139(in_vector[139]),
        .in_vector_140(in_vector[140]),
        .in_vector_141(in_vector[141]),
        .in_vector_142(in_vector[142]),
        .in_vector_143(in_vector[143]),
        .in_vector_144(in_vector[144]),
        .in_vector_145(in_vector[145]),
        .in_vector_146(in_vector[146]),
        .in_vector_147(in_vector[147]),
        .in_vector_148(in_vector[148]),
        .in_vector_149(in_vector[149]),
        .in_vector_150(in_vector[150]),
        .in_vector_151(in_vector[151]),
        .in_vector_152(in_vector[152]),
        .in_vector_153(in_vector[153]),
        .in_vector_154(in_vector[154]),
        .in_vector_155(in_vector[155]),
        .in_vector_156(in_vector[156]),
        .in_vector_157(in_vector[157]),
        .in_vector_158(in_vector[158]),
        .in_vector_159(in_vector[159]),
        .in_vector_160(in_vector[160]),
        .in_vector_161(in_vector[161]),
        .in_vector_162(in_vector[162]),
        .in_vector_163(in_vector[163]),
        .in_vector_164(in_vector[164]),
        .in_vector_165(in_vector[165]),
        .in_vector_166(in_vector[166]),
        .in_vector_167(in_vector[167]),
        .in_vector_168(in_vector[168]),
        .in_vector_169(in_vector[169]),
        .in_vector_170(in_vector[170]),
        .in_vector_171(in_vector[171]),
        .in_vector_172(in_vector[172]),
        .in_vector_173(in_vector[173]),
        .in_vector_174(in_vector[174]),
        .in_vector_175(in_vector[175]),
        .in_vector_176(in_vector[176]),
        .in_vector_177(in_vector[177]),
        .in_vector_178(in_vector[178]),
        .in_vector_179(in_vector[179]),
        .in_vector_180(in_vector[180]),
        .in_vector_181(in_vector[181]),
        .in_vector_182(in_vector[182]),
        .in_vector_183(in_vector[183]),
        .in_vector_184(in_vector[184]),
        .in_vector_185(in_vector[185]),
        .in_vector_186(in_vector[186]),
        .in_vector_187(in_vector[187]),
        .in_vector_188(in_vector[188]),
        .in_vector_189(in_vector[189]),
        .in_vector_190(in_vector[190]),
        .in_vector_191(in_vector[191]),
        .in_vector_192(in_vector[192]),
        .in_vector_193(in_vector[193]),
        .in_vector_194(in_vector[194]),
        .in_vector_195(in_vector[195]),
        .in_vector_196(in_vector[196]),
        .in_vector_197(in_vector[197]),
        .in_vector_198(in_vector[198]),
        .in_vector_199(in_vector[199]),
        .in_vector_200(in_vector[200]),
        .in_vector_201(in_vector[201]),
        .in_vector_202(in_vector[202]),
        .in_vector_203(in_vector[203]),
        .in_vector_204(in_vector[204]),
        .in_vector_205(in_vector[205]),
        .in_vector_206(in_vector[206]),
        .in_vector_207(in_vector[207]),
        .in_vector_208(in_vector[208]),
        .in_vector_209(in_vector[209]),
        .in_vector_210(in_vector[210]),
        .in_vector_211(in_vector[211]),
        .in_vector_212(in_vector[212]),
        .in_vector_213(in_vector[213]),
        .in_vector_214(in_vector[214]),
        .in_vector_215(in_vector[215]),
        .in_vector_216(in_vector[216]),
        .in_vector_217(in_vector[217]),
        .in_vector_218(in_vector[218]),
        .in_vector_219(in_vector[219]),
        .in_vector_220(in_vector[220]),
        .in_vector_221(in_vector[221]),
        .in_vector_222(in_vector[222]),
        .in_vector_223(in_vector[223]),
        .in_vector_224(in_vector[224]),
        .in_vector_225(in_vector[225]),
        .in_vector_226(in_vector[226]),
        .in_vector_227(in_vector[227]),
        .in_vector_228(in_vector[228]),
        .in_vector_229(in_vector[229]),
        .in_vector_230(in_vector[230]),
        .in_vector_231(in_vector[231]),
        .in_vector_232(in_vector[232]),
        .in_vector_233(in_vector[233]),
        .in_vector_234(in_vector[234]),
        .in_vector_235(in_vector[235]),
        .in_vector_236(in_vector[236]),
        .in_vector_237(in_vector[237]),
        .in_vector_238(in_vector[238]),
        .in_vector_239(in_vector[239]),
        .in_vector_240(in_vector[240]),
        .in_vector_241(in_vector[241]),
        .in_vector_242(in_vector[242]),
        .in_vector_243(in_vector[243]),
        .in_vector_244(in_vector[244]),
        .in_vector_245(in_vector[245]),
        .in_vector_246(in_vector[246]),
        .in_vector_247(in_vector[247]),
        .in_vector_248(in_vector[248]),
        .in_vector_249(in_vector[249]),
        .in_vector_250(in_vector[250]),
        .in_vector_251(in_vector[251]),
        .in_vector_252(in_vector[252]),
        .in_vector_253(in_vector[253]),
        .in_vector_254(in_vector[254]),
        .in_vector_255(in_vector[255]),
        .in_vector_256(in_vector[256]),
        .in_vector_257(in_vector[257]),
        .in_vector_258(in_vector[258]),
        .in_vector_259(in_vector[259]),
        .in_vector_260(in_vector[260]),
        .in_vector_261(in_vector[261]),
        .in_vector_262(in_vector[262]),
        .in_vector_263(in_vector[263]),
        .in_vector_264(in_vector[264]),
        .in_vector_265(in_vector[265]),
        .in_vector_266(in_vector[266]),
        .in_vector_267(in_vector[267]),
        .in_vector_268(in_vector[268]),
        .in_vector_269(in_vector[269]),
        .in_vector_270(in_vector[270]),
        .in_vector_271(in_vector[271]),
        .in_vector_272(in_vector[272]),
        .in_vector_273(in_vector[273]),
        .in_vector_274(in_vector[274]),
        .in_vector_275(in_vector[275]),
        .in_vector_276(in_vector[276]),
        .in_vector_277(in_vector[277]),
        .in_vector_278(in_vector[278]),
        .in_vector_279(in_vector[279]),
        .in_vector_280(in_vector[280]),
        .in_vector_281(in_vector[281]),
        .in_vector_282(in_vector[282]),
        .in_vector_283(in_vector[283]),
        .in_vector_284(in_vector[284]),
        .in_vector_285(in_vector[285]),
        .in_vector_286(in_vector[286]),
        .in_vector_287(in_vector[287]),
        .in_vector_288(in_vector[288]),
        .in_vector_289(in_vector[289]),
        .in_vector_290(in_vector[290]),
        .in_vector_291(in_vector[291]),
        .in_vector_292(in_vector[292]),
        .in_vector_293(in_vector[293]),
        .in_vector_294(in_vector[294]),
        .in_vector_295(in_vector[295]),
        .in_vector_296(in_vector[296]),
        .in_vector_297(in_vector[297]),
        .in_vector_298(in_vector[298]),
        .in_vector_299(in_vector[299]),
        .in_vector_300(in_vector[300]),
        .in_vector_301(in_vector[301]),
        .in_vector_302(in_vector[302]),
        .in_vector_303(in_vector[303]),
        .in_vector_304(in_vector[304]),
        .in_vector_305(in_vector[305]),
        .in_vector_306(in_vector[306]),
        .in_vector_307(in_vector[307]),
        .in_vector_308(in_vector[308]),
        .in_vector_309(in_vector[309]),
        .in_vector_310(in_vector[310]),
        .in_vector_311(in_vector[311]),
        .in_vector_312(in_vector[312]),
        .in_vector_313(in_vector[313]),
        .in_vector_314(in_vector[314]),
        .in_vector_315(in_vector[315]),
        .in_vector_316(in_vector[316]),
        .in_vector_317(in_vector[317]),
        .in_vector_318(in_vector[318]),
        .in_vector_319(in_vector[319]),
        .in_vector_320(in_vector[320]),
        .in_vector_321(in_vector[321]),
        .in_vector_322(in_vector[322]),
        .in_vector_323(in_vector[323]),
        .in_vector_324(in_vector[324]),
        .in_vector_325(in_vector[325]),
        .in_vector_326(in_vector[326]),
        .in_vector_327(in_vector[327]),
        .in_vector_328(in_vector[328]),
        .in_vector_329(in_vector[329]),
        .in_vector_330(in_vector[330]),
        .in_vector_331(in_vector[331]),
        .in_vector_332(in_vector[332]),
        .in_vector_333(in_vector[333]),
        .in_vector_334(in_vector[334]),
        .in_vector_335(in_vector[335]),
        .in_vector_336(in_vector[336]),
        .in_vector_337(in_vector[337]),
        .in_vector_338(in_vector[338]),
        .in_vector_339(in_vector[339]),
        .in_vector_340(in_vector[340]),
        .in_vector_341(in_vector[341]),
        .in_vector_342(in_vector[342]),
        .in_vector_343(in_vector[343]),
        .in_vector_344(in_vector[344]),
        .in_vector_345(in_vector[345]),
        .in_vector_346(in_vector[346]),
        .in_vector_347(in_vector[347]),
        .in_vector_348(in_vector[348]),
        .in_vector_349(in_vector[349]),
        .in_vector_350(in_vector[350]),
        .in_vector_351(in_vector[351]),
        .in_vector_352(in_vector[352]),
        .in_vector_353(in_vector[353]),
        .in_vector_354(in_vector[354]),
        .in_vector_355(in_vector[355]),
        .in_vector_356(in_vector[356]),
        .in_vector_357(in_vector[357]),
        .in_vector_358(in_vector[358]),
        .in_vector_359(in_vector[359]),
        .in_vector_360(in_vector[360]),
        .in_vector_361(in_vector[361]),
        .in_vector_362(in_vector[362]),
        .in_vector_363(in_vector[363]),
        .in_vector_364(in_vector[364]),
        .in_vector_365(in_vector[365]),
        .in_vector_366(in_vector[366]),
        .in_vector_367(in_vector[367]),
        .in_vector_368(in_vector[368]),
        .in_vector_369(in_vector[369]),
        .in_vector_370(in_vector[370]),
        .in_vector_371(in_vector[371]),
        .in_vector_372(in_vector[372]),
        .in_vector_373(in_vector[373]),
        .in_vector_374(in_vector[374]),
        .in_vector_375(in_vector[375]),
        .in_vector_376(in_vector[376]),
        .in_vector_377(in_vector[377]),
        .in_vector_378(in_vector[378]),
        .in_vector_379(in_vector[379]),
        .in_vector_380(in_vector[380]),
        .in_vector_381(in_vector[381]),
        .in_vector_382(in_vector[382]),
        .in_vector_383(in_vector[383]),
        .in_vector_384(in_vector[384]),
        .in_vector_385(in_vector[385]),
        .in_vector_386(in_vector[386]),
        .in_vector_387(in_vector[387]),
        .in_vector_388(in_vector[388]),
        .in_vector_389(in_vector[389]),
        .in_vector_390(in_vector[390]),
        .in_vector_391(in_vector[391]),
        .in_vector_392(in_vector[392]),
        .in_vector_393(in_vector[393]),
        .in_vector_394(in_vector[394]),
        .in_vector_395(in_vector[395]),
        .in_vector_396(in_vector[396]),
        .in_vector_397(in_vector[397]),
        .in_vector_398(in_vector[398]),
        .in_vector_399(in_vector[399]),
        .in_vector_400(in_vector[400]),
        .in_vector_401(in_vector[401]),
        .in_vector_402(in_vector[402]),
        .in_vector_403(in_vector[403]),
        .in_vector_404(in_vector[404]),
        .in_vector_405(in_vector[405]),
        .in_vector_406(in_vector[406]),
        .in_vector_407(in_vector[407]),
        .in_vector_408(in_vector[408]),
        .in_vector_409(in_vector[409]),
        .in_vector_410(in_vector[410]),
        .in_vector_411(in_vector[411]),
        .in_vector_412(in_vector[412]),
        .in_vector_413(in_vector[413]),
        .in_vector_414(in_vector[414]),
        .in_vector_415(in_vector[415]),
        .in_vector_416(in_vector[416]),
        .in_vector_417(in_vector[417]),
        .in_vector_418(in_vector[418]),
        .in_vector_419(in_vector[419]),
        .in_vector_420(in_vector[420]),
        .in_vector_421(in_vector[421]),
        .in_vector_422(in_vector[422]),
        .in_vector_423(in_vector[423]),
        .in_vector_424(in_vector[424]),
        .in_vector_425(in_vector[425]),
        .in_vector_426(in_vector[426]),
        .in_vector_427(in_vector[427]),
        .in_vector_428(in_vector[428]),
        .in_vector_429(in_vector[429]),
        .in_vector_430(in_vector[430]),
        .in_vector_431(in_vector[431]),
        .in_vector_432(in_vector[432]),
        .in_vector_433(in_vector[433]),
        .in_vector_434(in_vector[434]),
        .in_vector_435(in_vector[435]),
        .in_vector_436(in_vector[436]),
        .in_vector_437(in_vector[437]),
        .in_vector_438(in_vector[438]),
        .in_vector_439(in_vector[439]),
        .in_vector_440(in_vector[440]),
        .in_vector_441(in_vector[441]),
        .in_vector_442(in_vector[442]),
        .in_vector_443(in_vector[443]),
        .in_vector_444(in_vector[444]),
        .in_vector_445(in_vector[445]),
        .in_vector_446(in_vector[446]),
        .in_vector_447(in_vector[447]),
        .in_vector_448(in_vector[448]),
        .in_vector_449(in_vector[449]),
        .in_vector_450(in_vector[450]),
        .in_vector_451(in_vector[451]),
        .in_vector_452(in_vector[452]),
        .in_vector_453(in_vector[453]),
        .in_vector_454(in_vector[454]),
        .in_vector_455(in_vector[455]),
        .in_vector_456(in_vector[456]),
        .in_vector_457(in_vector[457]),
        .in_vector_458(in_vector[458]),
        .in_vector_459(in_vector[459]),
        .in_vector_460(in_vector[460]),
        .in_vector_461(in_vector[461]),
        .in_vector_462(in_vector[462]),
        .in_vector_463(in_vector[463]),
        .in_vector_464(in_vector[464]),
        .in_vector_465(in_vector[465]),
        .in_vector_466(in_vector[466]),
        .in_vector_467(in_vector[467]),
        .in_vector_468(in_vector[468]),
        .in_vector_469(in_vector[469]),
        .in_vector_470(in_vector[470]),
        .in_vector_471(in_vector[471]),
        .in_vector_472(in_vector[472]),
        .in_vector_473(in_vector[473]),
        .in_vector_474(in_vector[474]),
        .in_vector_475(in_vector[475]),
        .in_vector_476(in_vector[476]),
        .in_vector_477(in_vector[477]),
        .in_vector_478(in_vector[478]),
        .in_vector_479(in_vector[479]),
        .in_vector_480(in_vector[480]),
        .in_vector_481(in_vector[481]),
        .in_vector_482(in_vector[482]),
        .in_vector_483(in_vector[483]),
        .in_vector_484(in_vector[484]),
        .in_vector_485(in_vector[485]),
        .in_vector_486(in_vector[486]),
        .in_vector_487(in_vector[487]),
        .in_vector_488(in_vector[488]),
        .in_vector_489(in_vector[489]),
        .in_vector_490(in_vector[490]),
        .in_vector_491(in_vector[491]),
        .in_vector_492(in_vector[492]),
        .in_vector_493(in_vector[493]),
        .in_vector_494(in_vector[494]),
        .in_vector_495(in_vector[495]),
        .in_vector_496(in_vector[496]),
        .in_vector_497(in_vector[497]),
        .in_vector_498(in_vector[498]),
        .in_vector_499(in_vector[499]),
        .in_vector_500(in_vector[500]),
        .in_vector_501(in_vector[501]),
        .in_vector_502(in_vector[502]),
        .in_vector_503(in_vector[503]),
        .in_vector_504(in_vector[504]),
        .in_vector_505(in_vector[505]),
        .in_vector_506(in_vector[506]),
        .in_vector_507(in_vector[507]),
        .in_vector_508(in_vector[508]),
        .in_vector_509(in_vector[509]),
        .in_vector_510(in_vector[510]),
        .in_vector_511(in_vector[511]),
        .in_vector_512(in_vector[512]),
        .in_vector_513(in_vector[513]),
        .in_vector_514(in_vector[514]),
        .in_vector_515(in_vector[515]),
        .in_vector_516(in_vector[516]),
        .in_vector_517(in_vector[517]),
        .in_vector_518(in_vector[518]),
        .in_vector_519(in_vector[519]),
        .in_vector_520(in_vector[520]),
        .in_vector_521(in_vector[521]),
        .in_vector_522(in_vector[522]),
        .in_vector_523(in_vector[523]),
        .in_vector_524(in_vector[524]),
        .in_vector_525(in_vector[525]),
        .in_vector_526(in_vector[526]),
        .in_vector_527(in_vector[527]),
        .in_vector_528(in_vector[528]),
        .in_vector_529(in_vector[529]),
        .in_vector_530(in_vector[530]),
        .in_vector_531(in_vector[531]),
        .in_vector_532(in_vector[532]),
        .in_vector_533(in_vector[533]),
        .in_vector_534(in_vector[534]),
        .in_vector_535(in_vector[535]),
        .in_vector_536(in_vector[536]),
        .in_vector_537(in_vector[537]),
        .in_vector_538(in_vector[538]),
        .in_vector_539(in_vector[539]),
        .in_vector_540(in_vector[540]),
        .in_vector_541(in_vector[541]),
        .in_vector_542(in_vector[542]),
        .in_vector_543(in_vector[543]),
        .in_vector_544(in_vector[544]),
        .in_vector_545(in_vector[545]),
        .in_vector_546(in_vector[546]),
        .in_vector_547(in_vector[547]),
        .in_vector_548(in_vector[548]),
        .in_vector_549(in_vector[549]),
        .in_vector_550(in_vector[550]),
        .in_vector_551(in_vector[551]),
        .in_vector_552(in_vector[552]),
        .in_vector_553(in_vector[553]),
        .in_vector_554(in_vector[554]),
        .in_vector_555(in_vector[555]),
        .in_vector_556(in_vector[556]),
        .in_vector_557(in_vector[557]),
        .in_vector_558(in_vector[558]),
        .in_vector_559(in_vector[559]),
        .in_vector_560(in_vector[560]),
        .in_vector_561(in_vector[561]),
        .in_vector_562(in_vector[562]),
        .in_vector_563(in_vector[563]),
        .in_vector_564(in_vector[564]),
        .in_vector_565(in_vector[565]),
        .in_vector_566(in_vector[566]),
        .in_vector_567(in_vector[567]),
        .in_vector_568(in_vector[568]),
        .in_vector_569(in_vector[569]),
        .in_vector_570(in_vector[570]),
        .in_vector_571(in_vector[571]),
        .in_vector_572(in_vector[572]),
        .in_vector_573(in_vector[573]),
        .in_vector_574(in_vector[574]),
        .in_vector_575(in_vector[575]),
        .in_vector_576(in_vector[576]),
        .in_vector_577(in_vector[577]),
        .in_vector_578(in_vector[578]),
        .in_vector_579(in_vector[579]),
        .in_vector_580(in_vector[580]),
        .in_vector_581(in_vector[581]),
        .in_vector_582(in_vector[582]),
        .in_vector_583(in_vector[583]),
        .in_vector_584(in_vector[584]),
        .in_vector_585(in_vector[585]),
        .in_vector_586(in_vector[586]),
        .in_vector_587(in_vector[587]),
        .in_vector_588(in_vector[588]),
        .in_vector_589(in_vector[589]),
        .in_vector_590(in_vector[590]),
        .in_vector_591(in_vector[591]),
        .in_vector_592(in_vector[592]),
        .in_vector_593(in_vector[593]),
        .in_vector_594(in_vector[594]),
        .in_vector_595(in_vector[595]),
        .in_vector_596(in_vector[596]),
        .in_vector_597(in_vector[597]),
        .in_vector_598(in_vector[598]),
        .in_vector_599(in_vector[599]),
        .in_vector_600(in_vector[600]),
        .in_vector_601(in_vector[601]),
        .in_vector_602(in_vector[602]),
        .in_vector_603(in_vector[603]),
        .in_vector_604(in_vector[604]),
        .in_vector_605(in_vector[605]),
        .in_vector_606(in_vector[606]),
        .in_vector_607(in_vector[607]),
        .in_vector_608(in_vector[608]),
        .in_vector_609(in_vector[609]),
        .in_vector_610(in_vector[610]),
        .in_vector_611(in_vector[611]),
        .in_vector_612(in_vector[612]),
        .in_vector_613(in_vector[613]),
        .in_vector_614(in_vector[614]),
        .in_vector_615(in_vector[615]),
        .in_vector_616(in_vector[616]),
        .in_vector_617(in_vector[617]),
        .in_vector_618(in_vector[618]),
        .in_vector_619(in_vector[619]),
        .in_vector_620(in_vector[620]),
        .in_vector_621(in_vector[621]),
        .in_vector_622(in_vector[622]),
        .in_vector_623(in_vector[623]),
        .in_vector_624(in_vector[624]),
        .in_vector_625(in_vector[625]),
        .in_vector_626(in_vector[626]),
        .in_vector_627(in_vector[627]),
        .in_vector_628(in_vector[628]),
        .in_vector_629(in_vector[629]),
        .in_vector_630(in_vector[630]),
        .in_vector_631(in_vector[631]),
        .in_vector_632(in_vector[632]),
        .in_vector_633(in_vector[633]),
        .in_vector_634(in_vector[634]),
        .in_vector_635(in_vector[635]),
        .in_vector_636(in_vector[636]),
        .in_vector_637(in_vector[637]),
        .in_vector_638(in_vector[638]),
        .in_vector_639(in_vector[639]),
        .in_vector_640(in_vector[640]),
        .in_vector_641(in_vector[641]),
        .in_vector_642(in_vector[642]),
        .in_vector_643(in_vector[643]),
        .in_vector_644(in_vector[644]),
        .in_vector_645(in_vector[645]),
        .in_vector_646(in_vector[646]),
        .in_vector_647(in_vector[647]),
        .in_vector_648(in_vector[648]),
        .in_vector_649(in_vector[649]),
        .in_vector_650(in_vector[650]),
        .in_vector_651(in_vector[651]),
        .in_vector_652(in_vector[652]),
        .in_vector_653(in_vector[653]),
        .in_vector_654(in_vector[654]),
        .in_vector_655(in_vector[655]),
        .in_vector_656(in_vector[656]),
        .in_vector_657(in_vector[657]),
        .in_vector_658(in_vector[658]),
        .in_vector_659(in_vector[659]),
        .in_vector_660(in_vector[660]),
        .in_vector_661(in_vector[661]),
        .in_vector_662(in_vector[662]),
        .in_vector_663(in_vector[663]),
        .in_vector_664(in_vector[664]),
        .in_vector_665(in_vector[665]),
        .in_vector_666(in_vector[666]),
        .in_vector_667(in_vector[667]),
        .in_vector_668(in_vector[668]),
        .in_vector_669(in_vector[669]),
        .in_vector_670(in_vector[670]),
        .in_vector_671(in_vector[671]),
        .in_vector_672(in_vector[672]),
        .in_vector_673(in_vector[673]),
        .in_vector_674(in_vector[674]),
        .in_vector_675(in_vector[675]),
        .in_vector_676(in_vector[676]),
        .in_vector_677(in_vector[677]),
        .in_vector_678(in_vector[678]),
        .in_vector_679(in_vector[679]),
        .in_vector_680(in_vector[680]),
        .in_vector_681(in_vector[681]),
        .in_vector_682(in_vector[682]),
        .in_vector_683(in_vector[683]),
        .in_vector_684(in_vector[684]),
        .in_vector_685(in_vector[685]),
        .in_vector_686(in_vector[686]),
        .in_vector_687(in_vector[687]),
        .in_vector_688(in_vector[688]),
        .in_vector_689(in_vector[689]),
        .in_vector_690(in_vector[690]),
        .in_vector_691(in_vector[691]),
        .in_vector_692(in_vector[692]),
        .in_vector_693(in_vector[693]),
        .in_vector_694(in_vector[694]),
        .in_vector_695(in_vector[695]),
        .in_vector_696(in_vector[696]),
        .in_vector_697(in_vector[697]),
        .in_vector_698(in_vector[698]),
        .in_vector_699(in_vector[699]),
        .in_vector_700(in_vector[700]),
        .in_vector_701(in_vector[701]),
        .in_vector_702(in_vector[702]),
        .in_vector_703(in_vector[703]),
        .in_vector_704(in_vector[704]),
        .in_vector_705(in_vector[705]),
        .in_vector_706(in_vector[706]),
        .in_vector_707(in_vector[707]),
        .in_vector_708(in_vector[708]),
        .in_vector_709(in_vector[709]),
        .in_vector_710(in_vector[710]),
        .in_vector_711(in_vector[711]),
        .in_vector_712(in_vector[712]),
        .in_vector_713(in_vector[713]),
        .in_vector_714(in_vector[714]),
        .in_vector_715(in_vector[715]),
        .in_vector_716(in_vector[716]),
        .in_vector_717(in_vector[717]),
        .in_vector_718(in_vector[718]),
        .in_vector_719(in_vector[719]),
        .in_vector_720(in_vector[720]),
        .in_vector_721(in_vector[721]),
        .in_vector_722(in_vector[722]),
        .in_vector_723(in_vector[723]),
        .in_vector_724(in_vector[724]),
        .in_vector_725(in_vector[725]),
        .in_vector_726(in_vector[726]),
        .in_vector_727(in_vector[727]),
        .in_vector_728(in_vector[728]),
        .in_vector_729(in_vector[729]),
        .in_vector_730(in_vector[730]),
        .in_vector_731(in_vector[731]),
        .in_vector_732(in_vector[732]),
        .in_vector_733(in_vector[733]),
        .in_vector_734(in_vector[734]),
        .in_vector_735(in_vector[735]),
        .in_vector_736(in_vector[736]),
        .in_vector_737(in_vector[737]),
        .in_vector_738(in_vector[738]),
        .in_vector_739(in_vector[739]),
        .in_vector_740(in_vector[740]),
        .in_vector_741(in_vector[741]),
        .in_vector_742(in_vector[742]),
        .in_vector_743(in_vector[743]),
        .in_vector_744(in_vector[744]),
        .in_vector_745(in_vector[745]),
        .in_vector_746(in_vector[746]),
        .in_vector_747(in_vector[747]),
        .in_vector_748(in_vector[748]),
        .in_vector_749(in_vector[749]),
        .in_vector_750(in_vector[750]),
        .in_vector_751(in_vector[751]),
        .in_vector_752(in_vector[752]),
        .in_vector_753(in_vector[753]),
        .in_vector_754(in_vector[754]),
        .in_vector_755(in_vector[755]),
        .in_vector_756(in_vector[756]),
        .in_vector_757(in_vector[757]),
        .in_vector_758(in_vector[758]),
        .in_vector_759(in_vector[759]),
        .in_vector_760(in_vector[760]),
        .in_vector_761(in_vector[761]),
        .in_vector_762(in_vector[762]),
        .in_vector_763(in_vector[763]),
        .in_vector_764(in_vector[764]),
        .in_vector_765(in_vector[765]),
        .in_vector_766(in_vector[766]),
        .in_vector_767(in_vector[767]),
        .in_vector_768(in_vector[768]),
        .in_vector_769(in_vector[769]),
        .in_vector_770(in_vector[770]),
        .in_vector_771(in_vector[771]),
        .in_vector_772(in_vector[772]),
        .in_vector_773(in_vector[773]),
        .in_vector_774(in_vector[774]),
        .in_vector_775(in_vector[775]),
        .in_vector_776(in_vector[776]),
        .in_vector_777(in_vector[777]),
        .in_vector_778(in_vector[778]),
        .in_vector_779(in_vector[779]),
        .in_vector_780(in_vector[780]),
        .in_vector_781(in_vector[781]),
        .in_vector_782(in_vector[782]),
        .in_vector_783(in_vector[783]),
        .weight_matrix_0(weight_matrix_0),
        .weight_matrix_1(weight_matrix_1),
        .weight_matrix_2(weight_matrix_2),
        .weight_matrix_3(weight_matrix_3),
        .weight_matrix_4(weight_matrix_4),
        .weight_matrix_5(weight_matrix_5),
        .weight_matrix_6(weight_matrix_6),
        .weight_matrix_7(weight_matrix_7),
        .weight_matrix_8(weight_matrix_8),
        .output_vector(pe_result)
    );

    // Clock Generation
    always begin
        #5 clk = ~clk; // 100 MHz clock
    end

    // Initial Block to Stimulate Inputs
    initial begin
        // Initialize clock and reset
        clk = 0;
        rst = 1;  // Apply reset
        #10 rst = 0; // Release reset after 10ns
		
		

        // Set the 3x3 window of the image for the first MAC operation (Top-left corner of the image)
        in_vector[0] = 8'h00; //pixel at image0
        in_vector[1] = 8'h00; //pixel at image1
        in_vector[2] = 8'h00; //pixel at image2
        in_vector[3] = 8'h00; //pixel at image3
        in_vector[4] = 8'h00; //pixel at image4
        in_vector[5] = 8'h00; //pixel at image5
        in_vector[6] = 8'h00; //pixel at image6
        in_vector[7] = 8'h00; //pixel at image7
        in_vector[8] = 8'h00; //pixel at image8
        in_vector[9] = 8'h00; //pixel at image9
        in_vector[10] = 8'h00; //pixel at image10
        in_vector[11] = 8'h00; //pixel at image11
        in_vector[12] = 8'h00; //pixel at image12
        in_vector[13] = 8'h00; //pixel at image13
        in_vector[14] = 8'h00; //pixel at image14
        in_vector[15] = 8'h00; //pixel at image15
        in_vector[16] = 8'h00; //pixel at image16
        in_vector[17] = 8'h00; //pixel at image17
        in_vector[18] = 8'h00; //pixel at image18
        in_vector[19] = 8'h00; //pixel at image19
        in_vector[20] = 8'h00; //pixel at image20
        in_vector[21] = 8'h00; //pixel at image21
        in_vector[22] = 8'h00; //pixel at image22
        in_vector[23] = 8'h00; //pixel at image23
        in_vector[24] = 8'h00; //pixel at image24
        in_vector[25] = 8'h00; //pixel at image25
        in_vector[26] = 8'h00; //pixel at image26
        in_vector[27] = 8'h00; //pixel at image27
        in_vector[28] = 8'h00; //pixel at image28
        in_vector[29] = 8'h00; //pixel at image29
        in_vector[30] = 8'h00; //pixel at image30
        in_vector[31] = 8'h00; //pixel at image31
        in_vector[32] = 8'h00; //pixel at image32
        in_vector[33] = 8'h00; //pixel at image33
        in_vector[34] = 8'h00; //pixel at image34
        in_vector[35] = 8'h00; //pixel at image35
        in_vector[36] = 8'h00; //pixel at image36
        in_vector[37] = 8'h00; //pixel at image37
        in_vector[38] = 8'h00; //pixel at image38
        in_vector[39] = 8'h00; //pixel at image39
        in_vector[40] = 8'h00; //pixel at image40
        in_vector[41] = 8'h00; //pixel at image41
        in_vector[42] = 8'h00; //pixel at image42
        in_vector[43] = 8'h00; //pixel at image43
        in_vector[44] = 8'h00; //pixel at image44
        in_vector[45] = 8'h00; //pixel at image45
        in_vector[46] = 8'h00; //pixel at image46
        in_vector[47] = 8'h00; //pixel at image47
        in_vector[48] = 8'h00; //pixel at image48
        in_vector[49] = 8'h00; //pixel at image49
        in_vector[50] = 8'h00; //pixel at image50
        in_vector[51] = 8'h00; //pixel at image51
        in_vector[52] = 8'h00; //pixel at image52
        in_vector[53] = 8'h00; //pixel at image53
        in_vector[54] = 8'h00; //pixel at image54
        in_vector[55] = 8'h00; //pixel at image55
        in_vector[56] = 8'h00; //pixel at image56
        in_vector[57] = 8'h00; //pixel at image57
        in_vector[58] = 8'h00; //pixel at image58
        in_vector[59] = 8'h00; //pixel at image59
        in_vector[60] = 8'h00; //pixel at image60
        in_vector[61] = 8'h00; //pixel at image61
        in_vector[62] = 8'h00; //pixel at image62
        in_vector[63] = 8'h00; //pixel at image63
        in_vector[64] = 8'h00; //pixel at image64
        in_vector[65] = 8'h00; //pixel at image65
        in_vector[66] = 8'h00; //pixel at image66
        in_vector[67] = 8'h00; //pixel at image67
        in_vector[68] = 8'h00; //pixel at image68
        in_vector[69] = 8'h00; //pixel at image69
        in_vector[70] = 8'h00; //pixel at image70
        in_vector[71] = 8'h00; //pixel at image71
        in_vector[72] = 8'h00; //pixel at image72
        in_vector[73] = 8'h00; //pixel at image73
        in_vector[74] = 8'h00; //pixel at image74
        in_vector[75] = 8'h00; //pixel at image75
        in_vector[76] = 8'h00; //pixel at image76
        in_vector[77] = 8'h00; //pixel at image77
        in_vector[78] = 8'h00; //pixel at image78
        in_vector[79] = 8'h00; //pixel at image79
        in_vector[80] = 8'h00; //pixel at image80
        in_vector[81] = 8'h00; //pixel at image81
        in_vector[82] = 8'h00; //pixel at image82
        in_vector[83] = 8'h00; //pixel at image83
        in_vector[84] = 8'h00; //pixel at image84
        in_vector[85] = 8'h00; //pixel at image85
        in_vector[86] = 8'h00; //pixel at image86
        in_vector[87] = 8'h00; //pixel at image87
        in_vector[88] = 8'h00; //pixel at image88
        in_vector[89] = 8'h00; //pixel at image89
        in_vector[90] = 8'h00; //pixel at image90
        in_vector[91] = 8'h00; //pixel at image91
        in_vector[92] = 8'h00; //pixel at image92
        in_vector[93] = 8'h00; //pixel at image93
        in_vector[94] = 8'h00; //pixel at image94
        in_vector[95] = 8'h00; //pixel at image95
        in_vector[96] = 8'h00; //pixel at image96
        in_vector[97] = 8'h00; //pixel at image97
        in_vector[98] = 8'h00; //pixel at image98
        in_vector[99] = 8'h00; //pixel at image99
        in_vector[100] = 8'h00; //pixel at image100
        in_vector[101] = 8'h00; //pixel at image101
        in_vector[102] = 8'h00; //pixel at image102
        in_vector[103] = 8'h00; //pixel at image103
        in_vector[104] = 8'h00; //pixel at image104
        in_vector[105] = 8'h00; //pixel at image105
        in_vector[106] = 8'h00; //pixel at image106
        in_vector[107] = 8'h00; //pixel at image107
        in_vector[108] = 8'h00; //pixel at image108
        in_vector[109] = 8'h00; //pixel at image109
        in_vector[110] = 8'h00; //pixel at image110
        in_vector[111] = 8'h00; //pixel at image111
        in_vector[112] = 8'h00; //pixel at image112
        in_vector[113] = 8'h00; //pixel at image113
        in_vector[114] = 8'h00; //pixel at image114
        in_vector[115] = 8'h00; //pixel at image115
        in_vector[116] = 8'h00; //pixel at image116
        in_vector[117] = 8'h00; //pixel at image117
        in_vector[118] = 8'h00; //pixel at image118
        in_vector[119] = 8'h00; //pixel at image119
        in_vector[120] = 8'h00; //pixel at image120
        in_vector[121] = 8'h00; //pixel at image121
        in_vector[122] = 8'h00; //pixel at image122
        in_vector[123] = 8'h00; //pixel at image123
        in_vector[124] = 8'h00; //pixel at image124
        in_vector[125] = 8'h00; //pixel at image125
        in_vector[126] = 8'h00; //pixel at image126
        in_vector[127] = 8'h00; //pixel at image127
        in_vector[128] = 8'h00; //pixel at image128
        in_vector[129] = 8'h00; //pixel at image129
        in_vector[130] = 8'h00; //pixel at image130
        in_vector[131] = 8'h00; //pixel at image131
        in_vector[132] = 8'h00; //pixel at image132
        in_vector[133] = 8'h00; //pixel at image133
        in_vector[134] = 8'h00; //pixel at image134
        in_vector[135] = 8'h00; //pixel at image135
        in_vector[136] = 8'h00; //pixel at image136
        in_vector[137] = 8'h00; //pixel at image137
        in_vector[138] = 8'h00; //pixel at image138
        in_vector[139] = 8'h00; //pixel at image139
        in_vector[140] = 8'h00; //pixel at image140
        in_vector[141] = 8'h00; //pixel at image141
        in_vector[142] = 8'h00; //pixel at image142
        in_vector[143] = 8'h00; //pixel at image143
        in_vector[144] = 8'h00; //pixel at image144
        in_vector[145] = 8'h00; //pixel at image145
        in_vector[146] = 8'h00; //pixel at image146
        in_vector[147] = 8'h00; //pixel at image147
        in_vector[148] = 8'h00; //pixel at image148
        in_vector[149] = 8'h00; //pixel at image149
        in_vector[150] = 8'h00; //pixel at image150
        in_vector[151] = 8'h00; //pixel at image151
        in_vector[152] = 8'h03; //pixel at image152
        in_vector[153] = 8'h12; //pixel at image153
        in_vector[154] = 8'h12; //pixel at image154
        in_vector[155] = 8'h12; //pixel at image155
        in_vector[156] = 8'h7e; //pixel at image156
        in_vector[157] = 8'h88; //pixel at image157
        in_vector[158] = 8'haf; //pixel at image158
        in_vector[159] = 8'h1a; //pixel at image159
        in_vector[160] = 8'ha6; //pixel at image160
        in_vector[161] = 8'hff; //pixel at image161
        in_vector[162] = 8'hf7; //pixel at image162
        in_vector[163] = 8'h7f; //pixel at image163
        in_vector[164] = 8'h00; //pixel at image164
        in_vector[165] = 8'h00; //pixel at image165
        in_vector[166] = 8'h00; //pixel at image166
        in_vector[167] = 8'h00; //pixel at image167
        in_vector[168] = 8'h00; //pixel at image168
        in_vector[169] = 8'h00; //pixel at image169
        in_vector[170] = 8'h00; //pixel at image170
        in_vector[171] = 8'h00; //pixel at image171
        in_vector[172] = 8'h00; //pixel at image172
        in_vector[173] = 8'h00; //pixel at image173
        in_vector[174] = 8'h00; //pixel at image174
        in_vector[175] = 8'h00; //pixel at image175
        in_vector[176] = 8'h1e; //pixel at image176
        in_vector[177] = 8'h24; //pixel at image177
        in_vector[178] = 8'h5e; //pixel at image178
        in_vector[179] = 8'h9a; //pixel at image179
        in_vector[180] = 8'haa; //pixel at image180
        in_vector[181] = 8'hfd; //pixel at image181
        in_vector[182] = 8'hfd; //pixel at image182
        in_vector[183] = 8'hfd; //pixel at image183
        in_vector[184] = 8'hfd; //pixel at image184
        in_vector[185] = 8'hfd; //pixel at image185
        in_vector[186] = 8'he1; //pixel at image186
        in_vector[187] = 8'hac; //pixel at image187
        in_vector[188] = 8'hfd; //pixel at image188
        in_vector[189] = 8'hf2; //pixel at image189
        in_vector[190] = 8'hc3; //pixel at image190
        in_vector[191] = 8'h40; //pixel at image191
        in_vector[192] = 8'h00; //pixel at image192
        in_vector[193] = 8'h00; //pixel at image193
        in_vector[194] = 8'h00; //pixel at image194
        in_vector[195] = 8'h00; //pixel at image195
        in_vector[196] = 8'h00; //pixel at image196
        in_vector[197] = 8'h00; //pixel at image197
        in_vector[198] = 8'h00; //pixel at image198
        in_vector[199] = 8'h00; //pixel at image199
        in_vector[200] = 8'h00; //pixel at image200
        in_vector[201] = 8'h00; //pixel at image201
        in_vector[202] = 8'h00; //pixel at image202
        in_vector[203] = 8'h31; //pixel at image203
        in_vector[204] = 8'hee; //pixel at image204
        in_vector[205] = 8'hfd; //pixel at image205
        in_vector[206] = 8'hfd; //pixel at image206
        in_vector[207] = 8'hfd; //pixel at image207
        in_vector[208] = 8'hfd; //pixel at image208
        in_vector[209] = 8'hfd; //pixel at image209
        in_vector[210] = 8'hfd; //pixel at image210
        in_vector[211] = 8'hfd; //pixel at image211
        in_vector[212] = 8'hfd; //pixel at image212
        in_vector[213] = 8'hfb; //pixel at image213
        in_vector[214] = 8'h5d; //pixel at image214
        in_vector[215] = 8'h52; //pixel at image215
        in_vector[216] = 8'h52; //pixel at image216
        in_vector[217] = 8'h38; //pixel at image217
        in_vector[218] = 8'h27; //pixel at image218
        in_vector[219] = 8'h00; //pixel at image219
        in_vector[220] = 8'h00; //pixel at image220
        in_vector[221] = 8'h00; //pixel at image221
        in_vector[222] = 8'h00; //pixel at image222
        in_vector[223] = 8'h00; //pixel at image223
        in_vector[224] = 8'h00; //pixel at image224
        in_vector[225] = 8'h00; //pixel at image225
        in_vector[226] = 8'h00; //pixel at image226
        in_vector[227] = 8'h00; //pixel at image227
        in_vector[228] = 8'h00; //pixel at image228
        in_vector[229] = 8'h00; //pixel at image229
        in_vector[230] = 8'h00; //pixel at image230
        in_vector[231] = 8'h12; //pixel at image231
        in_vector[232] = 8'hdb; //pixel at image232
        in_vector[233] = 8'hfd; //pixel at image233
        in_vector[234] = 8'hfd; //pixel at image234
        in_vector[235] = 8'hfd; //pixel at image235
        in_vector[236] = 8'hfd; //pixel at image236
        in_vector[237] = 8'hfd; //pixel at image237
        in_vector[238] = 8'hc6; //pixel at image238
        in_vector[239] = 8'hb6; //pixel at image239
        in_vector[240] = 8'hf7; //pixel at image240
        in_vector[241] = 8'hf1; //pixel at image241
        in_vector[242] = 8'h00; //pixel at image242
        in_vector[243] = 8'h00; //pixel at image243
        in_vector[244] = 8'h00; //pixel at image244
        in_vector[245] = 8'h00; //pixel at image245
        in_vector[246] = 8'h00; //pixel at image246
        in_vector[247] = 8'h00; //pixel at image247
        in_vector[248] = 8'h00; //pixel at image248
        in_vector[249] = 8'h00; //pixel at image249
        in_vector[250] = 8'h00; //pixel at image250
        in_vector[251] = 8'h00; //pixel at image251
        in_vector[252] = 8'h00; //pixel at image252
        in_vector[253] = 8'h00; //pixel at image253
        in_vector[254] = 8'h00; //pixel at image254
        in_vector[255] = 8'h00; //pixel at image255
        in_vector[256] = 8'h00; //pixel at image256
        in_vector[257] = 8'h00; //pixel at image257
        in_vector[258] = 8'h00; //pixel at image258
        in_vector[259] = 8'h00; //pixel at image259
        in_vector[260] = 8'h50; //pixel at image260
        in_vector[261] = 8'h9c; //pixel at image261
        in_vector[262] = 8'h6b; //pixel at image262
        in_vector[263] = 8'hfd; //pixel at image263
        in_vector[264] = 8'hfd; //pixel at image264
        in_vector[265] = 8'hcd; //pixel at image265
        in_vector[266] = 8'h0b; //pixel at image266
        in_vector[267] = 8'h00; //pixel at image267
        in_vector[268] = 8'h2b; //pixel at image268
        in_vector[269] = 8'h9a; //pixel at image269
        in_vector[270] = 8'h00; //pixel at image270
        in_vector[271] = 8'h00; //pixel at image271
        in_vector[272] = 8'h00; //pixel at image272
        in_vector[273] = 8'h00; //pixel at image273
        in_vector[274] = 8'h00; //pixel at image274
        in_vector[275] = 8'h00; //pixel at image275
        in_vector[276] = 8'h00; //pixel at image276
        in_vector[277] = 8'h00; //pixel at image277
        in_vector[278] = 8'h00; //pixel at image278
        in_vector[279] = 8'h00; //pixel at image279
        in_vector[280] = 8'h00; //pixel at image280
        in_vector[281] = 8'h00; //pixel at image281
        in_vector[282] = 8'h00; //pixel at image282
        in_vector[283] = 8'h00; //pixel at image283
        in_vector[284] = 8'h00; //pixel at image284
        in_vector[285] = 8'h00; //pixel at image285
        in_vector[286] = 8'h00; //pixel at image286
        in_vector[287] = 8'h00; //pixel at image287
        in_vector[288] = 8'h00; //pixel at image288
        in_vector[289] = 8'h0e; //pixel at image289
        in_vector[290] = 8'h01; //pixel at image290
        in_vector[291] = 8'h9a; //pixel at image291
        in_vector[292] = 8'hfd; //pixel at image292
        in_vector[293] = 8'h5a; //pixel at image293
        in_vector[294] = 8'h00; //pixel at image294
        in_vector[295] = 8'h00; //pixel at image295
        in_vector[296] = 8'h00; //pixel at image296
        in_vector[297] = 8'h00; //pixel at image297
        in_vector[298] = 8'h00; //pixel at image298
        in_vector[299] = 8'h00; //pixel at image299
        in_vector[300] = 8'h00; //pixel at image300
        in_vector[301] = 8'h00; //pixel at image301
        in_vector[302] = 8'h00; //pixel at image302
        in_vector[303] = 8'h00; //pixel at image303
        in_vector[304] = 8'h00; //pixel at image304
        in_vector[305] = 8'h00; //pixel at image305
        in_vector[306] = 8'h00; //pixel at image306
        in_vector[307] = 8'h00; //pixel at image307
        in_vector[308] = 8'h00; //pixel at image308
        in_vector[309] = 8'h00; //pixel at image309
        in_vector[310] = 8'h00; //pixel at image310
        in_vector[311] = 8'h00; //pixel at image311
        in_vector[312] = 8'h00; //pixel at image312
        in_vector[313] = 8'h00; //pixel at image313
        in_vector[314] = 8'h00; //pixel at image314
        in_vector[315] = 8'h00; //pixel at image315
        in_vector[316] = 8'h00; //pixel at image316
        in_vector[317] = 8'h00; //pixel at image317
        in_vector[318] = 8'h00; //pixel at image318
        in_vector[319] = 8'h8b; //pixel at image319
        in_vector[320] = 8'hfd; //pixel at image320
        in_vector[321] = 8'hbe; //pixel at image321
        in_vector[322] = 8'h02; //pixel at image322
        in_vector[323] = 8'h00; //pixel at image323
        in_vector[324] = 8'h00; //pixel at image324
        in_vector[325] = 8'h00; //pixel at image325
        in_vector[326] = 8'h00; //pixel at image326
        in_vector[327] = 8'h00; //pixel at image327
        in_vector[328] = 8'h00; //pixel at image328
        in_vector[329] = 8'h00; //pixel at image329
        in_vector[330] = 8'h00; //pixel at image330
        in_vector[331] = 8'h00; //pixel at image331
        in_vector[332] = 8'h00; //pixel at image332
        in_vector[333] = 8'h00; //pixel at image333
        in_vector[334] = 8'h00; //pixel at image334
        in_vector[335] = 8'h00; //pixel at image335
        in_vector[336] = 8'h00; //pixel at image336
        in_vector[337] = 8'h00; //pixel at image337
        in_vector[338] = 8'h00; //pixel at image338
        in_vector[339] = 8'h00; //pixel at image339
        in_vector[340] = 8'h00; //pixel at image340
        in_vector[341] = 8'h00; //pixel at image341
        in_vector[342] = 8'h00; //pixel at image342
        in_vector[343] = 8'h00; //pixel at image343
        in_vector[344] = 8'h00; //pixel at image344
        in_vector[345] = 8'h00; //pixel at image345
        in_vector[346] = 8'h00; //pixel at image346
        in_vector[347] = 8'h0b; //pixel at image347
        in_vector[348] = 8'hbe; //pixel at image348
        in_vector[349] = 8'hfd; //pixel at image349
        in_vector[350] = 8'h46; //pixel at image350
        in_vector[351] = 8'h00; //pixel at image351
        in_vector[352] = 8'h00; //pixel at image352
        in_vector[353] = 8'h00; //pixel at image353
        in_vector[354] = 8'h00; //pixel at image354
        in_vector[355] = 8'h00; //pixel at image355
        in_vector[356] = 8'h00; //pixel at image356
        in_vector[357] = 8'h00; //pixel at image357
        in_vector[358] = 8'h00; //pixel at image358
        in_vector[359] = 8'h00; //pixel at image359
        in_vector[360] = 8'h00; //pixel at image360
        in_vector[361] = 8'h00; //pixel at image361
        in_vector[362] = 8'h00; //pixel at image362
        in_vector[363] = 8'h00; //pixel at image363
        in_vector[364] = 8'h00; //pixel at image364
        in_vector[365] = 8'h00; //pixel at image365
        in_vector[366] = 8'h00; //pixel at image366
        in_vector[367] = 8'h00; //pixel at image367
        in_vector[368] = 8'h00; //pixel at image368
        in_vector[369] = 8'h00; //pixel at image369
        in_vector[370] = 8'h00; //pixel at image370
        in_vector[371] = 8'h00; //pixel at image371
        in_vector[372] = 8'h00; //pixel at image372
        in_vector[373] = 8'h00; //pixel at image373
        in_vector[374] = 8'h00; //pixel at image374
        in_vector[375] = 8'h00; //pixel at image375
        in_vector[376] = 8'h23; //pixel at image376
        in_vector[377] = 8'hf1; //pixel at image377
        in_vector[378] = 8'he1; //pixel at image378
        in_vector[379] = 8'ha0; //pixel at image379
        in_vector[380] = 8'h6c; //pixel at image380
        in_vector[381] = 8'h01; //pixel at image381
        in_vector[382] = 8'h00; //pixel at image382
        in_vector[383] = 8'h00; //pixel at image383
        in_vector[384] = 8'h00; //pixel at image384
        in_vector[385] = 8'h00; //pixel at image385
        in_vector[386] = 8'h00; //pixel at image386
        in_vector[387] = 8'h00; //pixel at image387
        in_vector[388] = 8'h00; //pixel at image388
        in_vector[389] = 8'h00; //pixel at image389
        in_vector[390] = 8'h00; //pixel at image390
        in_vector[391] = 8'h00; //pixel at image391
        in_vector[392] = 8'h00; //pixel at image392
        in_vector[393] = 8'h00; //pixel at image393
        in_vector[394] = 8'h00; //pixel at image394
        in_vector[395] = 8'h00; //pixel at image395
        in_vector[396] = 8'h00; //pixel at image396
        in_vector[397] = 8'h00; //pixel at image397
        in_vector[398] = 8'h00; //pixel at image398
        in_vector[399] = 8'h00; //pixel at image399
        in_vector[400] = 8'h00; //pixel at image400
        in_vector[401] = 8'h00; //pixel at image401
        in_vector[402] = 8'h00; //pixel at image402
        in_vector[403] = 8'h00; //pixel at image403
        in_vector[404] = 8'h00; //pixel at image404
        in_vector[405] = 8'h51; //pixel at image405
        in_vector[406] = 8'hf0; //pixel at image406
        in_vector[407] = 8'hfd; //pixel at image407
        in_vector[408] = 8'hfd; //pixel at image408
        in_vector[409] = 8'h77; //pixel at image409
        in_vector[410] = 8'h19; //pixel at image410
        in_vector[411] = 8'h00; //pixel at image411
        in_vector[412] = 8'h00; //pixel at image412
        in_vector[413] = 8'h00; //pixel at image413
        in_vector[414] = 8'h00; //pixel at image414
        in_vector[415] = 8'h00; //pixel at image415
        in_vector[416] = 8'h00; //pixel at image416
        in_vector[417] = 8'h00; //pixel at image417
        in_vector[418] = 8'h00; //pixel at image418
        in_vector[419] = 8'h00; //pixel at image419
        in_vector[420] = 8'h00; //pixel at image420
        in_vector[421] = 8'h00; //pixel at image421
        in_vector[422] = 8'h00; //pixel at image422
        in_vector[423] = 8'h00; //pixel at image423
        in_vector[424] = 8'h00; //pixel at image424
        in_vector[425] = 8'h00; //pixel at image425
        in_vector[426] = 8'h00; //pixel at image426
        in_vector[427] = 8'h00; //pixel at image427
        in_vector[428] = 8'h00; //pixel at image428
        in_vector[429] = 8'h00; //pixel at image429
        in_vector[430] = 8'h00; //pixel at image430
        in_vector[431] = 8'h00; //pixel at image431
        in_vector[432] = 8'h00; //pixel at image432
        in_vector[433] = 8'h00; //pixel at image433
        in_vector[434] = 8'h2d; //pixel at image434
        in_vector[435] = 8'hba; //pixel at image435
        in_vector[436] = 8'hfd; //pixel at image436
        in_vector[437] = 8'hfd; //pixel at image437
        in_vector[438] = 8'h96; //pixel at image438
        in_vector[439] = 8'h1b; //pixel at image439
        in_vector[440] = 8'h00; //pixel at image440
        in_vector[441] = 8'h00; //pixel at image441
        in_vector[442] = 8'h00; //pixel at image442
        in_vector[443] = 8'h00; //pixel at image443
        in_vector[444] = 8'h00; //pixel at image444
        in_vector[445] = 8'h00; //pixel at image445
        in_vector[446] = 8'h00; //pixel at image446
        in_vector[447] = 8'h00; //pixel at image447
        in_vector[448] = 8'h00; //pixel at image448
        in_vector[449] = 8'h00; //pixel at image449
        in_vector[450] = 8'h00; //pixel at image450
        in_vector[451] = 8'h00; //pixel at image451
        in_vector[452] = 8'h00; //pixel at image452
        in_vector[453] = 8'h00; //pixel at image453
        in_vector[454] = 8'h00; //pixel at image454
        in_vector[455] = 8'h00; //pixel at image455
        in_vector[456] = 8'h00; //pixel at image456
        in_vector[457] = 8'h00; //pixel at image457
        in_vector[458] = 8'h00; //pixel at image458
        in_vector[459] = 8'h00; //pixel at image459
        in_vector[460] = 8'h00; //pixel at image460
        in_vector[461] = 8'h00; //pixel at image461
        in_vector[462] = 8'h00; //pixel at image462
        in_vector[463] = 8'h10; //pixel at image463
        in_vector[464] = 8'h5d; //pixel at image464
        in_vector[465] = 8'hfc; //pixel at image465
        in_vector[466] = 8'hfd; //pixel at image466
        in_vector[467] = 8'hbb; //pixel at image467
        in_vector[468] = 8'h00; //pixel at image468
        in_vector[469] = 8'h00; //pixel at image469
        in_vector[470] = 8'h00; //pixel at image470
        in_vector[471] = 8'h00; //pixel at image471
        in_vector[472] = 8'h00; //pixel at image472
        in_vector[473] = 8'h00; //pixel at image473
        in_vector[474] = 8'h00; //pixel at image474
        in_vector[475] = 8'h00; //pixel at image475
        in_vector[476] = 8'h00; //pixel at image476
        in_vector[477] = 8'h00; //pixel at image477
        in_vector[478] = 8'h00; //pixel at image478
        in_vector[479] = 8'h00; //pixel at image479
        in_vector[480] = 8'h00; //pixel at image480
        in_vector[481] = 8'h00; //pixel at image481
        in_vector[482] = 8'h00; //pixel at image482
        in_vector[483] = 8'h00; //pixel at image483
        in_vector[484] = 8'h00; //pixel at image484
        in_vector[485] = 8'h00; //pixel at image485
        in_vector[486] = 8'h00; //pixel at image486
        in_vector[487] = 8'h00; //pixel at image487
        in_vector[488] = 8'h00; //pixel at image488
        in_vector[489] = 8'h00; //pixel at image489
        in_vector[490] = 8'h00; //pixel at image490
        in_vector[491] = 8'h00; //pixel at image491
        in_vector[492] = 8'h00; //pixel at image492
        in_vector[493] = 8'hf9; //pixel at image493
        in_vector[494] = 8'hfd; //pixel at image494
        in_vector[495] = 8'hf9; //pixel at image495
        in_vector[496] = 8'h40; //pixel at image496
        in_vector[497] = 8'h00; //pixel at image497
        in_vector[498] = 8'h00; //pixel at image498
        in_vector[499] = 8'h00; //pixel at image499
        in_vector[500] = 8'h00; //pixel at image500
        in_vector[501] = 8'h00; //pixel at image501
        in_vector[502] = 8'h00; //pixel at image502
        in_vector[503] = 8'h00; //pixel at image503
        in_vector[504] = 8'h00; //pixel at image504
        in_vector[505] = 8'h00; //pixel at image505
        in_vector[506] = 8'h00; //pixel at image506
        in_vector[507] = 8'h00; //pixel at image507
        in_vector[508] = 8'h00; //pixel at image508
        in_vector[509] = 8'h00; //pixel at image509
        in_vector[510] = 8'h00; //pixel at image510
        in_vector[511] = 8'h00; //pixel at image511
        in_vector[512] = 8'h00; //pixel at image512
        in_vector[513] = 8'h00; //pixel at image513
        in_vector[514] = 8'h00; //pixel at image514
        in_vector[515] = 8'h00; //pixel at image515
        in_vector[516] = 8'h00; //pixel at image516
        in_vector[517] = 8'h00; //pixel at image517
        in_vector[518] = 8'h2e; //pixel at image518
        in_vector[519] = 8'h82; //pixel at image519
        in_vector[520] = 8'hb7; //pixel at image520
        in_vector[521] = 8'hfd; //pixel at image521
        in_vector[522] = 8'hfd; //pixel at image522
        in_vector[523] = 8'hcf; //pixel at image523
        in_vector[524] = 8'h02; //pixel at image524
        in_vector[525] = 8'h00; //pixel at image525
        in_vector[526] = 8'h00; //pixel at image526
        in_vector[527] = 8'h00; //pixel at image527
        in_vector[528] = 8'h00; //pixel at image528
        in_vector[529] = 8'h00; //pixel at image529
        in_vector[530] = 8'h00; //pixel at image530
        in_vector[531] = 8'h00; //pixel at image531
        in_vector[532] = 8'h00; //pixel at image532
        in_vector[533] = 8'h00; //pixel at image533
        in_vector[534] = 8'h00; //pixel at image534
        in_vector[535] = 8'h00; //pixel at image535
        in_vector[536] = 8'h00; //pixel at image536
        in_vector[537] = 8'h00; //pixel at image537
        in_vector[538] = 8'h00; //pixel at image538
        in_vector[539] = 8'h00; //pixel at image539
        in_vector[540] = 8'h00; //pixel at image540
        in_vector[541] = 8'h00; //pixel at image541
        in_vector[542] = 8'h00; //pixel at image542
        in_vector[543] = 8'h00; //pixel at image543
        in_vector[544] = 8'h27; //pixel at image544
        in_vector[545] = 8'h94; //pixel at image545
        in_vector[546] = 8'he5; //pixel at image546
        in_vector[547] = 8'hfd; //pixel at image547
        in_vector[548] = 8'hfd; //pixel at image548
        in_vector[549] = 8'hfd; //pixel at image549
        in_vector[550] = 8'hfa; //pixel at image550
        in_vector[551] = 8'hb6; //pixel at image551
        in_vector[552] = 8'h00; //pixel at image552
        in_vector[553] = 8'h00; //pixel at image553
        in_vector[554] = 8'h00; //pixel at image554
        in_vector[555] = 8'h00; //pixel at image555
        in_vector[556] = 8'h00; //pixel at image556
        in_vector[557] = 8'h00; //pixel at image557
        in_vector[558] = 8'h00; //pixel at image558
        in_vector[559] = 8'h00; //pixel at image559
        in_vector[560] = 8'h00; //pixel at image560
        in_vector[561] = 8'h00; //pixel at image561
        in_vector[562] = 8'h00; //pixel at image562
        in_vector[563] = 8'h00; //pixel at image563
        in_vector[564] = 8'h00; //pixel at image564
        in_vector[565] = 8'h00; //pixel at image565
        in_vector[566] = 8'h00; //pixel at image566
        in_vector[567] = 8'h00; //pixel at image567
        in_vector[568] = 8'h00; //pixel at image568
        in_vector[569] = 8'h00; //pixel at image569
        in_vector[570] = 8'h18; //pixel at image570
        in_vector[571] = 8'h72; //pixel at image571
        in_vector[572] = 8'hdd; //pixel at image572
        in_vector[573] = 8'hfd; //pixel at image573
        in_vector[574] = 8'hfd; //pixel at image574
        in_vector[575] = 8'hfd; //pixel at image575
        in_vector[576] = 8'hfd; //pixel at image576
        in_vector[577] = 8'hc9; //pixel at image577
        in_vector[578] = 8'h4e; //pixel at image578
        in_vector[579] = 8'h00; //pixel at image579
        in_vector[580] = 8'h00; //pixel at image580
        in_vector[581] = 8'h00; //pixel at image581
        in_vector[582] = 8'h00; //pixel at image582
        in_vector[583] = 8'h00; //pixel at image583
        in_vector[584] = 8'h00; //pixel at image584
        in_vector[585] = 8'h00; //pixel at image585
        in_vector[586] = 8'h00; //pixel at image586
        in_vector[587] = 8'h00; //pixel at image587
        in_vector[588] = 8'h00; //pixel at image588
        in_vector[589] = 8'h00; //pixel at image589
        in_vector[590] = 8'h00; //pixel at image590
        in_vector[591] = 8'h00; //pixel at image591
        in_vector[592] = 8'h00; //pixel at image592
        in_vector[593] = 8'h00; //pixel at image593
        in_vector[594] = 8'h00; //pixel at image594
        in_vector[595] = 8'h00; //pixel at image595
        in_vector[596] = 8'h17; //pixel at image596
        in_vector[597] = 8'h42; //pixel at image597
        in_vector[598] = 8'hd5; //pixel at image598
        in_vector[599] = 8'hfd; //pixel at image599
        in_vector[600] = 8'hfd; //pixel at image600
        in_vector[601] = 8'hfd; //pixel at image601
        in_vector[602] = 8'hfd; //pixel at image602
        in_vector[603] = 8'hc6; //pixel at image603
        in_vector[604] = 8'h51; //pixel at image604
        in_vector[605] = 8'h02; //pixel at image605
        in_vector[606] = 8'h00; //pixel at image606
        in_vector[607] = 8'h00; //pixel at image607
        in_vector[608] = 8'h00; //pixel at image608
        in_vector[609] = 8'h00; //pixel at image609
        in_vector[610] = 8'h00; //pixel at image610
        in_vector[611] = 8'h00; //pixel at image611
        in_vector[612] = 8'h00; //pixel at image612
        in_vector[613] = 8'h00; //pixel at image613
        in_vector[614] = 8'h00; //pixel at image614
        in_vector[615] = 8'h00; //pixel at image615
        in_vector[616] = 8'h00; //pixel at image616
        in_vector[617] = 8'h00; //pixel at image617
        in_vector[618] = 8'h00; //pixel at image618
        in_vector[619] = 8'h00; //pixel at image619
        in_vector[620] = 8'h00; //pixel at image620
        in_vector[621] = 8'h00; //pixel at image621
        in_vector[622] = 8'h12; //pixel at image622
        in_vector[623] = 8'hab; //pixel at image623
        in_vector[624] = 8'hdb; //pixel at image624
        in_vector[625] = 8'hfd; //pixel at image625
        in_vector[626] = 8'hfd; //pixel at image626
        in_vector[627] = 8'hfd; //pixel at image627
        in_vector[628] = 8'hfd; //pixel at image628
        in_vector[629] = 8'hc3; //pixel at image629
        in_vector[630] = 8'h50; //pixel at image630
        in_vector[631] = 8'h09; //pixel at image631
        in_vector[632] = 8'h00; //pixel at image632
        in_vector[633] = 8'h00; //pixel at image633
        in_vector[634] = 8'h00; //pixel at image634
        in_vector[635] = 8'h00; //pixel at image635
        in_vector[636] = 8'h00; //pixel at image636
        in_vector[637] = 8'h00; //pixel at image637
        in_vector[638] = 8'h00; //pixel at image638
        in_vector[639] = 8'h00; //pixel at image639
        in_vector[640] = 8'h00; //pixel at image640
        in_vector[641] = 8'h00; //pixel at image641
        in_vector[642] = 8'h00; //pixel at image642
        in_vector[643] = 8'h00; //pixel at image643
        in_vector[644] = 8'h00; //pixel at image644
        in_vector[645] = 8'h00; //pixel at image645
        in_vector[646] = 8'h00; //pixel at image646
        in_vector[647] = 8'h00; //pixel at image647
        in_vector[648] = 8'h37; //pixel at image648
        in_vector[649] = 8'hac; //pixel at image649
        in_vector[650] = 8'he2; //pixel at image650
        in_vector[651] = 8'hfd; //pixel at image651
        in_vector[652] = 8'hfd; //pixel at image652
        in_vector[653] = 8'hfd; //pixel at image653
        in_vector[654] = 8'hfd; //pixel at image654
        in_vector[655] = 8'hf4; //pixel at image655
        in_vector[656] = 8'h85; //pixel at image656
        in_vector[657] = 8'h0b; //pixel at image657
        in_vector[658] = 8'h00; //pixel at image658
        in_vector[659] = 8'h00; //pixel at image659
        in_vector[660] = 8'h00; //pixel at image660
        in_vector[661] = 8'h00; //pixel at image661
        in_vector[662] = 8'h00; //pixel at image662
        in_vector[663] = 8'h00; //pixel at image663
        in_vector[664] = 8'h00; //pixel at image664
        in_vector[665] = 8'h00; //pixel at image665
        in_vector[666] = 8'h00; //pixel at image666
        in_vector[667] = 8'h00; //pixel at image667
        in_vector[668] = 8'h00; //pixel at image668
        in_vector[669] = 8'h00; //pixel at image669
        in_vector[670] = 8'h00; //pixel at image670
        in_vector[671] = 8'h00; //pixel at image671
        in_vector[672] = 8'h00; //pixel at image672
        in_vector[673] = 8'h00; //pixel at image673
        in_vector[674] = 8'h00; //pixel at image674
        in_vector[675] = 8'h00; //pixel at image675
        in_vector[676] = 8'h88; //pixel at image676
        in_vector[677] = 8'hfd; //pixel at image677
        in_vector[678] = 8'hfd; //pixel at image678
        in_vector[679] = 8'hfd; //pixel at image679
        in_vector[680] = 8'hd4; //pixel at image680
        in_vector[681] = 8'h87; //pixel at image681
        in_vector[682] = 8'h84; //pixel at image682
        in_vector[683] = 8'h10; //pixel at image683
        in_vector[684] = 8'h00; //pixel at image684
        in_vector[685] = 8'h00; //pixel at image685
        in_vector[686] = 8'h00; //pixel at image686
        in_vector[687] = 8'h00; //pixel at image687
        in_vector[688] = 8'h00; //pixel at image688
        in_vector[689] = 8'h00; //pixel at image689
        in_vector[690] = 8'h00; //pixel at image690
        in_vector[691] = 8'h00; //pixel at image691
        in_vector[692] = 8'h00; //pixel at image692
        in_vector[693] = 8'h00; //pixel at image693
        in_vector[694] = 8'h00; //pixel at image694
        in_vector[695] = 8'h00; //pixel at image695
        in_vector[696] = 8'h00; //pixel at image696
        in_vector[697] = 8'h00; //pixel at image697
        in_vector[698] = 8'h00; //pixel at image698
        in_vector[699] = 8'h00; //pixel at image699
        in_vector[700] = 8'h00; //pixel at image700
        in_vector[701] = 8'h00; //pixel at image701
        in_vector[702] = 8'h00; //pixel at image702
        in_vector[703] = 8'h00; //pixel at image703
        in_vector[704] = 8'h00; //pixel at image704
        in_vector[705] = 8'h00; //pixel at image705
        in_vector[706] = 8'h00; //pixel at image706
        in_vector[707] = 8'h00; //pixel at image707
        in_vector[708] = 8'h00; //pixel at image708
        in_vector[709] = 8'h00; //pixel at image709
        in_vector[710] = 8'h00; //pixel at image710
        in_vector[711] = 8'h00; //pixel at image711
        in_vector[712] = 8'h00; //pixel at image712
        in_vector[713] = 8'h00; //pixel at image713
        in_vector[714] = 8'h00; //pixel at image714
        in_vector[715] = 8'h00; //pixel at image715
        in_vector[716] = 8'h00; //pixel at image716
        in_vector[717] = 8'h00; //pixel at image717
        in_vector[718] = 8'h00; //pixel at image718
        in_vector[719] = 8'h00; //pixel at image719
        in_vector[720] = 8'h00; //pixel at image720
        in_vector[721] = 8'h00; //pixel at image721
        in_vector[722] = 8'h00; //pixel at image722
        in_vector[723] = 8'h00; //pixel at image723
        in_vector[724] = 8'h00; //pixel at image724
        in_vector[725] = 8'h00; //pixel at image725
        in_vector[726] = 8'h00; //pixel at image726
        in_vector[727] = 8'h00; //pixel at image727
        in_vector[728] = 8'h00; //pixel at image728
        in_vector[729] = 8'h00; //pixel at image729
        in_vector[730] = 8'h00; //pixel at image730
        in_vector[731] = 8'h00; //pixel at image731
        in_vector[732] = 8'h00; //pixel at image732
        in_vector[733] = 8'h00; //pixel at image733
        in_vector[734] = 8'h00; //pixel at image734
        in_vector[735] = 8'h00; //pixel at image735
        in_vector[736] = 8'h00; //pixel at image736
        in_vector[737] = 8'h00; //pixel at image737
        in_vector[738] = 8'h00; //pixel at image738
        in_vector[739] = 8'h00; //pixel at image739
        in_vector[740] = 8'h00; //pixel at image740
        in_vector[741] = 8'h00; //pixel at image741
        in_vector[742] = 8'h00; //pixel at image742
        in_vector[743] = 8'h00; //pixel at image743
        in_vector[744] = 8'h00; //pixel at image744
        in_vector[745] = 8'h00; //pixel at image745
        in_vector[746] = 8'h00; //pixel at image746
        in_vector[747] = 8'h00; //pixel at image747
        in_vector[748] = 8'h00; //pixel at image748
        in_vector[749] = 8'h00; //pixel at image749
        in_vector[750] = 8'h00; //pixel at image750
        in_vector[751] = 8'h00; //pixel at image751
        in_vector[752] = 8'h00; //pixel at image752
        in_vector[753] = 8'h00; //pixel at image753
        in_vector[754] = 8'h00; //pixel at image754
        in_vector[755] = 8'h00; //pixel at image755
        in_vector[756] = 8'h00; //pixel at image756
        in_vector[757] = 8'h00; //pixel at image757
        in_vector[758] = 8'h00; //pixel at image758
        in_vector[759] = 8'h00; //pixel at image759
        in_vector[760] = 8'h00; //pixel at image760
        in_vector[761] = 8'h00; //pixel at image761
        in_vector[762] = 8'h00; //pixel at image762
        in_vector[763] = 8'h00; //pixel at image763
        in_vector[764] = 8'h00; //pixel at image764
        in_vector[765] = 8'h00; //pixel at image765
        in_vector[766] = 8'h00; //pixel at image766
        in_vector[767] = 8'h00; //pixel at image767
        in_vector[768] = 8'h00; //pixel at image768
        in_vector[769] = 8'h00; //pixel at image769
        in_vector[770] = 8'h00; //pixel at image770
        in_vector[771] = 8'h00; //pixel at image771
        in_vector[772] = 8'h00; //pixel at image772
        in_vector[773] = 8'h00; //pixel at image773
        in_vector[774] = 8'h00; //pixel at image774
        in_vector[775] = 8'h00; //pixel at image775
        in_vector[776] = 8'h00; //pixel at image776
        in_vector[777] = 8'h00; //pixel at image777
        in_vector[778] = 8'h00; //pixel at image778
        in_vector[779] = 8'h00; //pixel at image779
        in_vector[780] = 8'h00; //pixel at image780
        in_vector[781] = 8'h00; //pixel at image781
        in_vector[782] = 8'h00; //pixel at image782
        in_vector[783] = 8'h00; //pixel at image783

        // Sobel-Y top-edge, encoded as uint8 (two's complement for negatives)
        weight_matrix_0 = 8'hFF; // -1
        weight_matrix_1 = 8'hFE; // -2
        weight_matrix_2 = 8'hFF; // -1
        
        weight_matrix_3 = 8'h00; //  0
        weight_matrix_4 = 8'h00; //  0
        weight_matrix_5 = 8'h00; //  0
        
        weight_matrix_6 = 8'h01; // +1
        weight_matrix_7 = 8'h02; // +2
        weight_matrix_8 = 8'h01; // +1

        // Apply stimulus for each sliding window of 3x3 and store results
        // ---- Capture 26x26 outputs correctly (advance time) ----
        // PE starts at window (0,0); capture that first
		@(posedge clk);
        #1;
        output_vector[0][0] = pe_result;

        // Capture remaining windows: one per posedge clk
        for (k = 1; k < (26*26); k = k + 1) begin
            @(posedge clk);
            #1; // allow PE output to settle after internal window update
            row = k / 26;
            col = k % 26;
            output_vector[row][col] = pe_result;
        end


        // Monitor the output for the first pixel (top-left corner of the output map)
        $monitor("At time %t, pe_result = %0d", $time, pe_result); // Expect the sum result

                $display("Capture done at time %0t", $time);
        $display("out[0][0]=%0d out[0][1]=%0d out[0][2]=%0d", output_vector[0][0], output_vector[0][1], output_vector[0][2]);
        $display("out[1][0]=%0d out[1][1]=%0d out[1][2]=%0d", output_vector[1][0], output_vector[1][1], output_vector[1][2]);
        $display("out[25][25]=%0d", output_vector[25][25]);

// Finish simulation after observing the result
        #1000;
        $finish;
    end

     integer f;
     integer count;
     
     initial begin
       f = $fopen("verilog_out_u16.txt","w");
       count = 0;
     end
     
     always @(posedge clk) begin
       if (!rst) begin
         count = count + 1;
     
         if (count > 1) begin
           $fwrite(f, "%0d\n", pe_result);  // use your connected output signal
         end
     
         if (count == 676+1) begin
           $fclose(f);
           $finish;
         end
       end
     end


endmodule
