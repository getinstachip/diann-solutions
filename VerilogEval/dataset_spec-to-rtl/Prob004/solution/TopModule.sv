module TopModule (
    input  [31:0] in,
    output [31:0] out
);

    ByteReverser byte_reverser_inst (
        .data_in(in),
        .data_out(out)
    );

endmodule