module XOR_Module (
    input  wire x,
    input  wire y,
    output wire xor_out
);
    assign xor_out = x ^ y;
endmodule