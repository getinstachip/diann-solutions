module ORGate (
    input in1,
    input in2,
    output or_out
);
    assign or_out = in1 | in2;
endmodule