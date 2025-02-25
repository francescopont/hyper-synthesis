dtmc

module example
    s : [0..12] init 0;

    [] (s=0) -> 0.5 : (s'=1) + 0.5 : (s'=2);
    [] (s=1) -> 0.5 : (s'=12) + 0.5 : (s'=3);
    [] (s=2) -> (s'=4);

endmodule


rewards
	[] s=2 : 1;
endrewards

label "target" = s=3 | s=4;

