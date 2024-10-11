module AckController (
    input  wire clk,
    input  wire reset,
    input  wire done,
    input  wire ack,
    output reg  reset_search
);

    typedef enum logic [1:0] {
        SEARCH,
        WAIT_ACK
    } state_t;

    state_t state, next_state;

    // State register
    always_ff @(posedge clk) begin
        if (reset) begin
            state        <= SEARCH;
            reset_search <= 1'b0;
        end else begin
            state <= next_state;
        end
    end

    // Next state logic
    always_comb begin
        case (state)
            SEARCH: begin
                if (done)
                    next_state = WAIT_ACK;
                else
                    next_state = SEARCH;
            end
            WAIT_ACK: begin
                if (ack)
                    next_state = SEARCH;
                else
                    next_state = WAIT_ACK;
            end
            default: next_state = SEARCH;
        endcase
    end

    // Output logic
    always_comb begin
        case (state)
            SEARCH: reset_search = done;
            WAIT_ACK: reset_search = 1'b1;
            default: reset_search = 1'b0;
        endcase
    end

endmodule