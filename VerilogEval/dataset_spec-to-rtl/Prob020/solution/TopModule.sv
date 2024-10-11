module TopModule (
    input  [1:0] A,
    input  [1:0] B,
    output       z
);

    // Instantiate Comparator submodule
    Comparator Comparator_inst (
        .A(A),
        .B(B),
        .z(z)
    );

endmodule