module ShiftRegister (
    input clk,
    input reset,
    input shift_enable,
    input [3:0] data_in,
    output reg [3:0] data_out
);

    always @(posedge clk) begin
        if (reset) begin
            data_out <= 4'b0000;
        end
        else if (shift_enable) begin
            data_out <= {data_out[2:0], data_in[0]};
        end
    end

endmodule