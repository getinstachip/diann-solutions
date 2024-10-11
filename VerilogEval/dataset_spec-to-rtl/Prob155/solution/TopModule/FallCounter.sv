module FallCounter (
    input  wire clk,
    input  wire enable,
    input  wire reset,
    output reg  fall_done
);

    // 5-bit counter to count clock cycles up to 21
    reg [4:0] count;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            count      <= 5'd0;
            fall_done  <= 1'b0;
        end
        else if (enable) begin
            if (count >= 5'd20) begin
                fall_done <= 1'b1;
            end
            else begin
                count <= count + 1;
                // Set fall_done when count exceeds 20
                fall_done <= (count + 1 > 5'd20) ? 1'b1 : 1'b0;
            end
        end
        else begin
            fall_done <= 1'b0;
        end
    end

endmodule