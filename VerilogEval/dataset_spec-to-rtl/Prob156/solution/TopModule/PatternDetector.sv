module PatternDetector (
    input wire clk,
    input wire reset,
    input wire data,
    output reg pattern_detected
);

    reg [3:0] shift_reg;

    always @(posedge clk) begin
        if (reset) begin
            shift_reg <= 4'b0000;
            pattern_detected <= 1'b0;
        end else begin
            shift_reg <= {shift_reg[2:0], data};
            if (shift_reg == 4'b1101)
                pattern_detected <= 1'b1;
            else
                pattern_detected <= 1'b0;
        end
    end

endmodule